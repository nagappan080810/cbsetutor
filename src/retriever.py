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
        question:       str,
        class_filter:   str  = None,
        subject_filter: str  = None,
        difficulty:     str  = None,   # "easy" | "medium" | "hard" | "mixed" | None
        prefer_exercise: bool = False  # True = boost end-of-chapter / Figure-It-Out content
    ) -> list:
        """
        Returns list of dicts:
        {
          "doc":        LangChain Document,
          "score":      float  (raw CrossEncoder score),
          "confidence": float  (0-100 percentage),
          "label":      str    ("High" / "Medium" / "Low" / "Very Low")
        }

        NOTE: Pure embedding similarity systematically under-ranks short,
        terse content like NCERT "Figure It Out" boxes and end-of-chapter
        exercises, because they don't share much vocabulary with a
        natural-language question. We correct for this below using the
        rich metadata written by ingest.py (content_type, bloom_level,
        question_location) — without this correction, retrieval silently
        skips exactly the harder / activity-based content.
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

        # Step 1 — vector search (retrieve MORE candidates than before so
        # low-similarity-but-high-value content like Figure It Out boxes
        # still has a chance to be pulled in before re-ranking)
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
            console.print(
                f"[yellow]⚠ No results found for class={class_filter!r} "
                f"subject={subject_filter!r}. Check that PDFs for this "
                f"class/subject were actually ingested (metadata 'class' "
                f"and 'subject' fields must match exactly).[/yellow]"
            )
            return []

        # Defensive check — candidates exist but have no usable text.
        # This can happen if ingest produced empty-content chunks (e.g. a
        # PDF page that failed text extraction but still got a Document).
        # Passing empty strings to the reranker silently degrades scores
        # rather than erroring, so we filter them out explicitly here.
        non_empty_candidates = [
            doc for doc in candidates if doc.page_content and doc.page_content.strip()
        ]
        if len(non_empty_candidates) < len(candidates):
            console.print(
                f"[yellow]⚠ Dropped {len(candidates) - len(non_empty_candidates)} "
                f"candidate(s) with empty page_content[/yellow]"
            )
        candidates = non_empty_candidates

        if not candidates:
            console.print(
                "[red]⚠ All retrieved candidates had empty content — "
                "this PDF's text was not extracted properly during ingest. "
                "Re-run ingest.py for this class/subject.[/red]"
            )
            return []

        # Step 2 — re-rank with CrossEncoder
        pairs = [[question, doc.page_content] for doc in candidates]

        if not pairs:
            # Should be unreachable given the guards above, but fail loudly
            # rather than calling predict([]) and silently getting back an
            # empty score array.
            console.print(
                "[red]⚠ Internal error: candidates non-empty but pairs is "
                "empty. Aborting re-rank.[/red]"
            )
            return []

        # Step 2 — re-rank with CrossEncoder
        # NOTE: pairs MUST be a list of [str, str] — verified below, since a
        # CrossEncoder silently produces empty/garbage output if any item in
        # the batch is not a plain string (e.g. None, a LangChain Document,
        # or a non-str type slipping through from a bad ingest chunk).
        bad_pairs = [
            (i, q, d) for i, (q, d) in enumerate(pairs)
            if not isinstance(q, str) or not isinstance(d, str)
        ]
        if bad_pairs:
            console.print(
                f"[red]⚠ Found {len(bad_pairs)} pair(s) with non-string "
                f"content — this silently breaks CrossEncoder.predict(). "
                f"First bad pair at index {bad_pairs[0][0]}: "
                f"q_type={type(bad_pairs[0][1]).__name__} "
                f"d_type={type(bad_pairs[0][2]).__name__}[/red]"
            )
            # Coerce to string rather than failing outright — keeps the
            # pipeline alive while surfacing the warning above
            pairs = [[str(q), str(d)] for q, d in pairs]

        console.print(
            f"[dim]  Calling reranker.predict() with {len(pairs)} pairs "
            f"(pair[0] types: {type(pairs[0][0]).__name__}, "
            f"{type(pairs[0][1]).__name__})[/dim]"
        )

        # Call predict() with an explicit batch_size — some sentence-
        # transformers versions have a bug/regression where the default
        # internal batching silently returns an empty result for certain
        # torch/transformers version combos. Forcing batch_size=1 (slower
        # but maximally safe) isolates whether batching itself is the cause.
        raw = self.reranker.predict(
            pairs,
            convert_to_numpy=True,
            batch_size=int(os.getenv("RERANKER_BATCH_SIZE", 8)),
            show_progress_bar=False,
        )

        console.print(
            f"[dim]  predict() returned: type={type(raw).__name__} "
            f"repr={raw!r}[/dim]"
        )

        # If predict() came back empty even with explicit batch_size, retry
        # once with batch_size=1 — this isolates a batching-specific bug
        # from a model/install-level failure.
        if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
            console.print(
                "[yellow]⚠ predict() returned empty with batch_size="
                f"{os.getenv('RERANKER_BATCH_SIZE', 8)}. Retrying with "
                "batch_size=1 to isolate a batching bug...[/yellow]"
            )
            raw = self.reranker.predict(
                pairs,
                convert_to_numpy=True,
                batch_size=1,
                show_progress_bar=False,
            )
            console.print(
                f"[dim]  Retry predict() returned: type={type(raw).__name__} "
                f"repr={raw!r}[/dim]"
            )

        # Diagnostic: print exactly what predict() handed back BEFORE any
        # processing, so we can tell apart "returned empty array" vs
        # "returned None" vs "returned wrong shape" vs "flatten() ate it"
        import numpy as np
        raw_type  = type(raw).__name__
        raw_shape = getattr(raw, "shape", "N/A (no .shape attr)")
        raw_len   = len(raw) if hasattr(raw, "__len__") else "N/A"
        console.print(
            f"[dim]  predict() raw output → type={raw_type} "
            f"shape={raw_shape} len={raw_len}[/dim]"
        )

        if raw is None:
            console.print(
                "[red]⚠ predict() returned None. This usually means the "
                "CrossEncoder model failed to load correctly, or a "
                "sentence-transformers version incompatibility. Try: "
                "pip install -U sentence-transformers torch[/red]"
            )
            return []

        # Normalise to a numpy array regardless of what predict() handed back
        # (covers: python list, list of tensors, 0-d array, etc.)
        try:
            raw_arr = np.asarray(raw, dtype=float)
        except Exception as e:
            console.print(
                f"[red]⚠ Could not convert predict() output to a numpy "
                f"array: {e}. Raw value was: {raw!r}[/red]"
            )
            return []

        if raw_arr.size == 0:
            console.print(
                f"[red]⚠ predict() returned an array with 0 elements for "
                f"{len(pairs)} input pairs. This is almost always caused by "
                f"one of:\n"
                f"   1. A version mismatch between sentence-transformers and "
                f"torch/transformers — try: pip install -U "
                f"sentence-transformers\n"
                f"   2. The reranker model failed silently during __init__ "
                f"(check for a warning earlier in the logs when "
                f"CrossEncoder(...) was constructed)\n"
                f"   3. Every chunk's text was reduced to empty after "
                f"tokenizer truncation (extremely unlikely unless "
                f"page_content contains only whitespace/control chars)\n"
                f"[/red]"
            )
            return []

        # predict() can return shape (N,) or (N,1) depending on the model.
        # Flatten to a guaranteed 1-D array then convert to plain Python floats.
        scores = raw_arr.flatten().tolist()   # always [f1, f2, ...] never [[f1],[f2],...]

        if not scores:
            console.print(
                f"[red]⚠ Reranker returned no scores for {len(pairs)} pairs — "
                f"this should not happen. Check the CrossEncoder model load "
                f"and pairs content.[/red]"
            )
            return []

        # Step 3 — compute confidence scores
        confidences = self._score_to_confidence(scores)

        # Step 4 — metadata-aware boost
        # Pure CrossEncoder score under-ranks short activity/exercise content,
        # so we apply a deterministic boost based on metadata. HOWEVER this
        # boost must only ever apply to candidates that the CrossEncoder
        # already judged as topically relevant — otherwise a structurally
        # interesting but topically WRONG chunk (e.g. an unrelated MCQ block
        # or activity box from a different section) gets promoted purely
        # because it matches a metadata tag, regardless of relevance.
        # This was a real bug: a percentages-example chunk was outranking a
        # genuinely relevant squares-of-numbers chunk because it happened to
        # be tagged content_type="activity" and received the +1.2 boost
        # unconditionally.
        BLOOM_FOR_DIFFICULTY = {
            "easy":   {"remember", "understand"},
            "medium": {"understand", "apply", "analyse"},
            "hard":   {"analyse", "evaluate"},
            "mixed":  {"remember", "understand", "apply", "analyse", "evaluate"},
        }
        target_blooms = BLOOM_FOR_DIFFICULTY.get(difficulty, None)

        # Relevance floor: only candidates scoring at or above the median of
        # this batch's raw CrossEncoder scores are eligible for metadata
        # boosts. This keeps the boost's purpose (rescue terse-but-relevant
        # content from being under-ranked) without letting it rescue
        # genuinely irrelevant content just because of a structural tag.
        sorted_scores = sorted(scores, reverse=True)
        relevance_floor = sorted_scores[len(sorted_scores) // 2] if sorted_scores else 0.0

        boosted_scores = []
        for raw_score, doc in zip(scores, candidates):
            meta  = doc.metadata
            boost = 0.0

            # Only apply ANY metadata boost if this candidate already
            # cleared the relevance floor — i.e. it's in the more-relevant
            # half of this batch on raw semantic/lexical grounds alone.
            eligible_for_boost = raw_score >= relevance_floor

            if eligible_for_boost:
                # Bloom level match — only applied if a difficulty was requested
                if target_blooms and meta.get("bloom_level") in target_blooms:
                    boost += 1.5   # added to the raw logit scale, not the % scale

                # Figure It Out / activity / higher-order content boost
                ctype = meta.get("content_type", "")
                loc   = meta.get("question_location", "")
                if ctype == "activity":
                    boost += 1.2   # NCERT "Figure It Out" boxes are tagged 'activity'
                if loc == "exercise" and prefer_exercise:
                    boost += 1.0
                if meta.get("bloom_level") == "evaluate":
                    boost += 0.8   # highest-order content gets a standing boost

                # Mild penalty for in-text discussion prompts when looking for
                # exam-style content (keeps them available, just not dominant)
                if loc == "intext" and prefer_exercise:
                    boost -= 0.6

            boosted_scores.append(raw_score + boost)

        console.print(
            f"[dim]  Relevance floor (median raw score): {relevance_floor:.3f}[/dim]"
        )
        console.print(
            f"[dim]  Raw scores: {[round(s,2) for s in scores]}[/dim]"
        )
        console.print(
            f"[dim]  Boosted:    {[round(s,2) for s in boosted_scores]}[/dim]"
        )

        # Step 5 — sort by BOOSTED score descending, keep top K
        combined = sorted(
            zip(boosted_scores, scores, confidences, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        top = combined[:self.top_k_rerank]

        results = []
        for boosted, raw_score, conf, doc in top:
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
            doc     = r["doc"]
            ctype   = doc.metadata.get("content_type", "?")
            console.print(
                f"  [{color}]{r['label']:9s} {r['confidence']:5.1f}%[/{color}] "
                f"[dim]{bar}[/dim] "
                f"[magenta]{ctype:12s}[/magenta] "
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

    @staticmethod
    def is_context_sufficient(results: list, min_confidence: float = 40.0) -> bool:
        """
        Decide whether retrieved context is strong enough to answer from
        confidently. If the top result's confidence is below threshold,
        the LLM is much more likely to fall back on outside knowledge
        (i.e. hallucinate) rather than admit the textbook doesn't cover it.

        Callers should check this BEFORE invoking the LLM, and short-circuit
        with an explicit "not found in textbook" message instead of asking
        the LLM to answer from weak context.
        """
        if not results:
            return False
        return results[0]["confidence"] >= min_confidence
