import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from rich.console import Console

load_dotenv()
console = Console()

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CBSERetriever:
    def __init__(self):
        self.embeddings  = self._get_embeddings()
        self.vectorstore = Chroma(
            persist_directory=os.getenv("CHROMA_DIR", "./chroma_db"),
            embedding_function=self.embeddings
        )
        console.print("[dim]Loading re-ranker model...[/dim]")
        self.reranker       = CrossEncoder(RERANKER_MODEL)
        self.top_k_retrieve = int(os.getenv("TOP_K_RETRIEVE", 20))
        self.top_k_rerank   = int(os.getenv("TOP_K_RERANK", 5))

    def _get_embeddings(self):
        provider = os.getenv("EMBED_PROVIDER", "ollama")
        console.print(f"[dim]Retriever embedding provider: {provider}[/dim]")

        if provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            model = os.getenv("EMBED_MODEL", "nomic-embed-text")
            console.print(f"[dim]Using Ollama model: {model} (768 dims)[/dim]")
            return OllamaEmbeddings(
                model=model,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
        elif provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            model = os.getenv("EMBED_MODEL", "text-embedding-004")
            console.print(f"[dim]Using Google model: {model}[/dim]")
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                task_type="retrieval_query"
            )
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            model = os.getenv(
                "EMBED_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            console.print(f"[dim]Using HuggingFace model: {model} (384 dims)[/dim]")
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            raise ValueError(
                f"Unknown EMBED_PROVIDER: {provider}. "
                "Use 'ollama', 'google', or 'huggingface' in .env"
            )

    @staticmethod
    def _score_to_confidence(scores: list) -> list:
        """
        Convert raw CrossEncoder scores to 0-100 confidence percentages.

        CrossEncoder scores are unbounded logits (can be negative or > 1).
        We apply sigmoid to map them to 0-1, then scale to 0-100.

        sigmoid(x) = 1 / (1 + e^(-x))
        - score  0  → 50%
        - score  2  → 88%
        - score  4  → 98%
        - score -2  → 12%
        """
        import math
        confidences = []
        for s in scores:
            sigmoid = 1.0 / (1.0 + math.exp(-float(s)))
            confidences.append(round(sigmoid * 100, 1))
        return confidences

    @staticmethod
    def _confidence_label(pct: float) -> str:
        """Return a human-readable label for the confidence percentage."""
        if pct >= 85: return "High"
        if pct >= 60: return "Medium"
        if pct >= 40: return "Low"
        return "Very Low"

    def retrieve_and_rerank(
        self,
        question:      str,
        class_filter:  str = None,
        subject_filter: str = None
    ) -> list:
        """
        Returns list of dicts:
        {
          "doc":        LangChain Document,
          "score":      float  (raw CrossEncoder score),
          "confidence": float  (0-100 percentage),
          "label":      str    ("High" / "Medium" / "Low" / "Very Low")
        }
        """
        # Build metadata filter
        where_filter = {}
        if class_filter and subject_filter:
            where_filter = {
                "$and": [
                    {"class":   {"$eq": class_filter}},
                    {"subject": {"$eq": subject_filter}}
                ]
            }
        elif class_filter:
            where_filter = {"class": {"$eq": class_filter}}
        elif subject_filter:
            where_filter = {"subject": {"$eq": subject_filter}}

        # Step 1 — vector search
        try:
            if where_filter:
                candidates = self.vectorstore.similarity_search(
                    question,
                    k=self.top_k_retrieve,
                    filter=where_filter
                )
            else:
                candidates = self.vectorstore.similarity_search(
                    question,
                    k=self.top_k_retrieve
                )
        except Exception as e:
            console.print(
                f"[yellow]Filter search failed ({e}), trying without filter...[/yellow]"
            )
            try:
                all_candidates = self.vectorstore.similarity_search(
                    question,
                    k=self.top_k_retrieve * 3
                )
                candidates = [
                    doc for doc in all_candidates
                    if (class_filter  is None or doc.metadata.get("class")   == class_filter)
                    and (subject_filter is None or doc.metadata.get("subject") == subject_filter)
                ][:self.top_k_retrieve]
                console.print(
                    f"[dim]  Fallback manual filter: {len(candidates)} candidates[/dim]"
                )
            except Exception as e2:
                console.print(f"[red]Search failed completely: {e2}[/red]")
                return []

        if not candidates:
            console.print("[yellow]No results found for selected class/subject.[/yellow]")
            return []

        # Step 2 — re-rank with CrossEncoder
        pairs  = [[question, doc.page_content] for doc in candidates]
        raw       = self.reranker.predict(pairs, convert_to_numpy=True)
        # predict() can return shape (N,) or (N,1) depending on the model.
        # Flatten to a guaranteed 1-D array then convert to plain Python floats.
        scores    = raw.flatten().tolist()   # always [f1, f2, ...] never [[f1],[f2],...]
        console.print(f"scores value {scores}")

        # Step 3 — compute confidence scores
        confidences = self._score_to_confidence(scores)
        console.print(f"scores value {confidences}")

        # Step 4 — sort by score descending, keep top K
        combined = sorted(
            zip(scores, confidences, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        top = combined[:self.top_k_rerank]

        results = []
        for raw_score, conf, doc in top:
            results.append({
                "doc":        doc,
                "score":      round(raw_score, 4),
                "confidence": conf,
                "label":      self._confidence_label(conf)
            })

        # Log to terminal
        console.print(
            f"[dim]  Retrieved {len(candidates)} → re-ranked → top {len(results)}[/dim]"
        )
        for r in results:
            bar_len = int(r["confidence"] / 10)
            bar     = "█" * bar_len + "░" * (10 - bar_len)
            color   = {"High": "green", "Medium": "yellow",
                       "Low": "red", "Very Low": "red"}.get(r["label"], "white")
            console.print(
                f"  [{color}]{r['label']:9s} {r['confidence']:5.1f}%[/{color}] "
                f"[dim]{bar}[/dim] "
                f"[white]{doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')}[/white]"
            )

        return results

    def get_top_docs(self, results: list):
        """Helper — extract just the Document objects from results."""
        return [r["doc"] for r in results]

    def overall_confidence(self, results: list) -> dict:
        """
        Compute an overall confidence score for the full answer.
        Uses weighted average of top result scores.
        """
        if not results:
            return {"score": 0.0, "label": "No Results", "color": "red"}

        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        total_w = 0.0
        total_s = 0.0
        for i, r in enumerate(results):
            w        = weights[i] if i < len(weights) else 0.1
            total_s += r["confidence"] * w
            total_w += w

        overall = round(total_s / total_w, 1) if total_w else 0.0
        label   = self._confidence_label(overall)
        color   = {"High": "#22c55e", "Medium": "#eab308",
                   "Low": "#f97316", "Very Low": "#ef4444"}.get(label, "#94a3b8")
        return {"score": overall, "label": label, "color": color}
