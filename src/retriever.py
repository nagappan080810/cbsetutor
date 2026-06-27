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
        """
        Must return the EXACT same embedding model used in ingest.py.
        Both read from the same .env so they always stay in sync.
        """
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

    def retrieve_and_rerank(
        self,
        question: str,
        class_filter: str = None,
        subject_filter: str = None
    ) -> list:

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

        # Step 1 — filtered vector search
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
            console.print(f"[red]Search error: {e}[/red]")
            return []

        if not candidates:
            console.print(
                "[yellow]No results found for selected class/subject.[/yellow]"
            )
            return []

        # Step 2 — re-rank using CrossEncoder
        pairs  = [[question, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        ranked   = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        top_docs = [doc for _, doc in ranked[:self.top_k_rerank]]

        console.print(
            f"[dim]  Retrieved {len(candidates)} → "
            f"re-ranked → top {len(top_docs)}[/dim]"
        )
        return top_docs