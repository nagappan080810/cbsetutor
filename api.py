"""
CBSE RAG - FastAPI Server

Pages (served from static/):
  GET /           → Chat UI
  GET /upload     → Upload PDF UI
  GET /quiz       → Quiz UI

API:
  GET  /health
  GET  /api/classes
  GET  /api/subjects/{class_name}
  GET  /api/files/{class_name}/{subject}
  POST /api/ask                         → SSE streaming answer + confidence
  POST /api/upload                      → Upload PDF + ingest
  POST /api/extract-questions           → Extract questions from PDF or HTML
  GET  /api/images/{filename}
  GET  /api/images
  GET  /api/ingest/status
"""

import os
import random
import re
import json
import math
import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich.console import Console

console = Console()
load_dotenv()

app = FastAPI(title="CBSE RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR        = os.getenv("DATA_DIR",        "./data")
CHROMA_DIR      = os.getenv("CHROMA_DIR",      "./chroma_db")
IMAGE_STORE_DIR = os.getenv("IMAGE_STORE_DIR", "./image_store")
TRACKER_FILE    = "ingest_tracker.json"

SUPPORTED_CLASSES  = ["class_8", "class_9", "class_10"]
SUPPORTED_SUBJECTS = [
    "mathematics", "science", "socialscience",
    "english", "hindi", "kannada", "tamil", "sanskrit"
]

LANGUAGE_INSTRUCTIONS = {
    "hindi":          "हिंदी में जवाब दें।",
    "kannada":        "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
    "tamil":          "தமிழில் பதில் சொல்லுங்கள்.",
    "sanskrit":       "संस्कृते उत्तरं देहि।",
    "mathematics":    "Respond in English with clear numbered step-by-step solutions.",
    "science":        "Respond in English with clear explanations.",
    "socialscience": "Respond in English.",
    "english":        "Respond in English.",
}

ANSWER_MODE_INSTRUCTIONS = {
    "detailed": """
Answer in detail:
- Explain the concept thoroughly
- For maths: show every step numbered (Step 1, Step 2...)
- Define all formulas and terms before using them
- End with a summary or key takeaway
""",
    "brief": """
Answer in exactly 30 to 40 words.
- Be precise and clear, no bullet points
- Include only the most important point
""",
    "one_line": """
Answer in a single sentence of no more than 30 words.
- Give only the direct answer, nothing else
""",
    "mcq": """
This is a Multiple Choice Question.
- First clearly state which option is correct (e.g. "The correct answer is (B)")
- Explain in 2-3 sentences WHY that option is correct
- Briefly explain why the other options are wrong
""",
    "true_false": """
This is a True or False question.
- First state clearly: TRUE or FALSE
- Give a clear reason in 2-3 sentences explaining why
- Reference the relevant concept from the textbook
""",
}

BASE_PROMPT = """You are an expert CBSE tutor for {class_name}, subject: {subject}.

{language_instruction}

Use ONLY the context below from NCERT textbooks to answer.

GROUNDING RULES:
- Do NOT invent facts, data, dates, or formulas that are not present in the
  CONTEXT below, even if you recall them from general knowledge.
- However, if the CONTEXT gives you a formula, rule, or method (e.g. the
  mirror/lens formula, a percentage/ratio rule, an algebraic identity), you
  MAY and SHOULD apply that formula to the exact numbers given in the
  QUESTION, even if this specific numeric example is not written verbatim
  in the textbook. Applying a textbook formula to new numbers is normal
  problem-solving, not hallucination -- only decline to answer when the
  underlying CONCEPT or FORMULA itself is missing from the CONTEXT, never
  merely because the specific numbers in the question don't appear there.
- For ANY arithmetic, algebra, or numeric result -- including rearranging
  a formula from the context to solve for an unknown -- ALWAYS call the
  `calculate` tool to get the exact value instead of computing it mentally.
- If the CONTEXT truly doesn't cover the relevant concept/formula at all,
  say so clearly instead of guessing.

ANSWER FORMAT:
{answer_mode_instruction}

General rules:
- Use simple language for a school student
- Use markdown: **bold** for key terms, ``` for formulas

---
CONTEXT:
{context}

---
QUESTION: {question}

ANSWER:"""

# ── Fallback prompt: used only when retrieval confidence is too low to
# ground the answer. Forcing BASE_PROMPT's strict "ONLY use context" rules
# onto irrelevant/empty context produces either a refusal or an LLM that
# keeps reaching for the `calculate` tool trying to reconcile numbers that
# aren't actually in the context -- which is what caused the retry-stall on
# self-contained problems (e.g. "verify this trig identity") that have no
# matching textbook passage to retrieve, by design. This lets the model
# answer from its own subject knowledge instead, clearly labelled as such.
GENERAL_KNOWLEDGE_PROMPT = """You are an expert CBSE tutor for {class_name}, subject: {subject}.

{language_instruction}

No sufficiently relevant passage was found in the NCERT textbook corpus for
this question -- expected for self-contained problems (e.g. "verify this
identity", "solve for x") that don't correspond to any specific textbook
passage, not necessarily a sign something is wrong.

Answer using your own subject knowledge instead, following these rules:
- Stay strictly within the CBSE {class_name} syllabus for {subject} -- do not
  use methods, notation, or content from a different grade level.
- For ANY arithmetic, algebra, or numeric result, ALWAYS call the
  `calculate` tool with a single self-contained expression instead of
  computing mentally. If a calculate call errors, do not retry the same
  broken form -- rephrase once as a single expression, or otherwise proceed
  with careful manual working.

ANSWER FORMAT:
{answer_mode_instruction}

General rules:
- Use simple language for a school student
- Use markdown: **bold** for key terms, ``` for formulas

---
QUESTION: {question}

ANSWER:"""

MIN_CONFIDENCE_FOR_GROUNDED_ANSWER = float(os.getenv("MIN_CONFIDENCE_FOR_GROUNDED_ANSWER", 35.0))

# ── Shared RAG chain (retriever + LLM, provider selection lives in one place) ──
from src.rag_chain import get_rag_chain
# CONCEPT GRAPH HOOK (1/2): import. Delete this line + the two call sites
# marked "CONCEPT GRAPH HOOK" below to fully remove the feature.
from src import concept_graph
from src.html_question_extractor import extract_questions_from_html, guess_question_type

# ── Request models ────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question:    str     = Field(..., min_length=3)
    class_name:  str     = Field(..., example="class_10")
    subject:     str     = Field(..., example="mathematics")
    max_context: int     = Field(2000, ge=500, le=5000)
    answer_mode: Literal["detailed","brief","one_line","mcq","true_false"] = "detailed"
    topics:      list[str] = Field(default_factory=list)   # optional topic hints (e.g. a teacher tagging an uploaded paper in quiz.html) used to widen/ground retrieval alongside the question text itself

class WorksheetRequest(BaseModel):
    class_name:   str        = Field(..., example="class_10")
    subject:      str        = Field(..., example="mathematics")
    topics:       list[str]  = Field(..., min_length=1)
    difficulty:   Literal["easy","medium","hard","mixed"] = "medium"
    question_types: list[Literal["mcq","short","truefalse","fillblank","long","assertreason"]] = ["mcq","short"]
    num_questions: int       = Field(10, ge=3, le=30)
    max_context:   int       = Field(3000, ge=500, le=6000)
    extra_instructions: str  = Field("", max_length=500)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_tracker() -> dict:
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE) as f:
            return json.load(f)
    return {}

def load_image_metadata() -> dict:
    meta = os.path.join(IMAGE_STORE_DIR, "metadata.json")
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f)
    return {}

def find_relevant_images(class_name, subject, source_docs, max_images=3):
    metadata   = load_image_metadata()
    all_images = metadata.get("images", [])
    if not all_images:
        return []

    referenced = [
        {"source": d.metadata.get("source",""), "page": d.metadata.get("page",-1)}
        for d in source_docs
        if d.metadata.get("source") and d.metadata.get("page",-1) >= 0
    ]

    scored = []
    for img in all_images:
        if img["class"] != class_name or img["subject"] != subject:
            continue
        score = 0
        for ref in referenced:
            if img["source_pdf"] == ref["source"]:
                diff = abs(img["page"] - ref["page"])
                if   diff == 0: score += 10
                elif diff == 1: score += 6
                elif diff == 2: score += 3
                elif diff <= 5: score += 1
        if img.get("has_drawings"): score += 2
        if score > 0: scored.append((score, img))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [img for _, img in scored[:max_images]]

def _extract_pdf_text(pdf_path: str) -> str:
    """fitz first (better layout handling), pypdf fallback. Shared by
    /api/extract-questions and the concept-graph ingestion hook so there's
    one implementation instead of two copies drifting apart."""
    text = ""
    try:
        import fitz
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.mupdf_display_warnings(False)
        doc = fitz.open(pdf_path)
        for page in doc:
            try: text += page.get_text("text") + "\n"
            except: pass
        doc.close()
    except Exception:
        import pypdf
        reader = pypdf.PdfReader(pdf_path, strict=False)
        for page in reader.pages:
            try: text += (page.extract_text() or "") + "\n"
            except: pass
    return text

def score_to_confidence(raw_score: float) -> float:
    """Sigmoid transform of CrossEncoder score → 0-100%."""
    sigmoid = 1.0 / (1.0 + math.exp(-float(raw_score)))
    return round(sigmoid * 100, 1)

def confidence_label(pct: float) -> str:
    if pct >= 85: return "High"
    if pct >= 60: return "Medium"
    if pct >= 40: return "Low"
    return "Very Low"

def confidence_color(pct: float) -> str:
    if pct >= 85: return "#22c55e"
    if pct >= 60: return "#eab308"
    if pct >= 40: return "#f97316"
    return "#ef4444"

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# 45s was an initial guess, not a measured value. Multi-part questions can
# legitimately need 3-4 sequential LLM round-trips (initial call -> tool
# result -> possibly another tool call -> final synthesis), and large NIM
# models can take 10-15s+ per round-trip on shared/free tiers -- so 45s
# total was sometimes just not enough time for genuinely correct, unstuck
# generation. Raise via LLM_STREAM_TIMEOUT_SECONDS if your provider is
# consistently slower than this default; the batching guidance in the
# `calculate` tool docstring (see rag_chain.py) reduces how many
# round-trips are needed in the first place, which matters more than the
# raw timeout value for multi-part questions.
LLM_STREAM_TIMEOUT_SECONDS = float(os.getenv("LLM_STREAM_TIMEOUT_SECONDS", 90.0))

async def _iter_with_timeout(agen, timeout: float = LLM_STREAM_TIMEOUT_SECONDS):
    """
    Wrap an async generator so that if any single step stalls for longer
    than `timeout` seconds, we raise asyncio.TimeoutError instead of hanging
    silently. This matters specifically for astream_with_tools(): a tool
    round-trip (e.g. a `calculate` call mid-generation) means a fresh LLM
    provider request gets made mid-stream, and if that particular request
    stalls or the provider drops the connection without an error, the caller
    previously saw nothing at all -- no more tokens, no error event, just an
    indefinitely "stuck" request with no feedback in the UI.
    """
    it = agen.__aiter__()
    while True:
        try:
            item = await asyncio.wait_for(it.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        yield item


class ThinkTagFilter:
    """
    Streaming filter that strips <think>...</think> reasoning blocks (some
    models, including the NVIDIA NIM models used here, emit these before
    their real answer) from a token stream -- correctly handling the tags
    being split across arbitrary chunk boundaries.

    Why the previous approach (`re.sub(r"<think>.*", "", token,
    flags=re.DOTALL)` applied independently to each streamed token) was
    broken: it can only strip content that lands in the SAME chunk as the
    literal "<think>" text. In practice, streamed tokens are small
    (sometimes single words/subwords) and a model's reasoning block spans
    many chunks -- so that regex reliably caught the opening tag (turning
    that one chunk to "") but had no memory of still being "inside" a think
    block for every subsequent chunk. Since those later chunks don't
    individually contain the literal string "<think>", the regex found
    nothing to substitute and passed them through completely unfiltered:
    raw chain-of-thought reasoning leaked straight into the visible answer.
    Worse, on questions where the model reasons for a long time before
    ever emitting closing remarks, wall-clock time spent generating that
    (invisible-to-filter but not invisible-to-timer) reasoning is what
    burns through the 45s budget before any real answer text arrives --
    which is why longer "verify this identity" questions were hit hardest.

    This class instead tracks open/close state explicitly across calls to
    feed(), and holds back only the minimum trailing buffer needed to
    detect a tag split across a chunk boundary (not the whole hidden
    reasoning block, so memory stays bounded regardless of how long the
    model reasons for).
    """
    _OPEN  = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self._buffer = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        out = []
        while True:
            tag = self._CLOSE if self._inside else self._OPEN
            idx = self._buffer.find(tag)
            if idx == -1:
                # No complete tag in the buffer yet. Hold back just enough
                # trailing characters in case they're the start of a tag
                # that continues in the next chunk; anything before that
                # is safe to flush now (if we're not currently hidden
                # inside a think block).
                keep = max(len(self._OPEN), len(self._CLOSE)) - 1
                safe_len = max(0, len(self._buffer) - keep)
                if not self._inside:
                    out.append(self._buffer[:safe_len])
                self._buffer = self._buffer[safe_len:]
                break
            if not self._inside:
                out.append(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(tag):]
            self._inside = not self._inside
        return "".join(out)

    def flush(self) -> str:
        """Call once after the stream ends to release any safely-held-back
        trailing text that turned out not to be a split tag after all."""
        remaining = self._buffer if not self._inside else ""
        self._buffer = ""
        return remaining


# ── SSE generator ─────────────────────────────────────────────────────────────
async def answer_stream(req: AskRequest) -> AsyncGenerator[str, None]:
    if req.class_name not in SUPPORTED_CLASSES:
        yield sse({"type":"error","message":f"Invalid class: {req.class_name}"}); return
    if req.subject not in SUPPORTED_SUBJECTS:
        yield sse({"type":"error","message":f"Invalid subject: {req.subject}"}); return

    yield sse({"type":"status","message":"Searching NCERT textbooks…"})
    await asyncio.sleep(0)

    # Retrieve + re-rank (returns list of dicts)
    try:
        retriever = get_rag_chain().retriever
        results   = retriever.retrieve_and_rerank(
            req.question,
            class_filter=req.class_name,
            subject_filter=req.subject
        )

        # Widen with topic-based retrieval if the caller supplied topics
        # (e.g. a teacher tagging an uploaded question paper in quiz.html
        # before answering it). An extracted question's exact wording can
        # be noisy -- OCR artefacts, exam-style phrasing, paraphrasing --
        # and may not embed close to the textbook's own wording even when
        # the concept is covered well. A topic name like "Reflection of
        # Light" usually retrieves the right section directly, so merging
        # that pool in (deduped by source+page) raises the confidence floor
        # instead of leaving the answer to rely on the question text alone.
        if req.topics:
            seen_keys = {
                f"{r['doc'].metadata.get('source','')}:{r['doc'].metadata.get('page','')}"
                for r in results
            }
            for topic in req.topics:
                topic_results = retriever.retrieve_and_rerank(
                    topic, class_filter=req.class_name, subject_filter=req.subject
                )
                for r in topic_results:
                    key = f"{r['doc'].metadata.get('source','')}:{r['doc'].metadata.get('page','')}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        results.append(r)
            results.sort(key=lambda r: r["confidence"], reverse=True)
            results = results[:8]   # keep the merged pool bounded
    except Exception as e:
        yield sse({"type":"error","message":f"Retrieval error: {e}"}); return

    if not results:
        yield sse({"type":"error","message":"No relevant content found. Make sure PDFs are ingested."}); return

    # Extract docs
    top_docs = [r["doc"] for r in results]

    # Compute overall confidence (weighted average)
    console.print(f"answer_stream results {results}")
    weights  = [1.0, 0.8, 0.6, 0.4, 0.2]
    total_w  = sum(weights[i] if i < len(weights) else 0.1 for i in range(len(results)))
    total_s  = sum(
        results[i]["confidence"] * (weights[i] if i < len(weights) else 0.1)
        for i in range(len(results))
    )
    console.print(f" results {total_w} {total_s}")
    overall_pct   = round(total_s / total_w, 1) if total_w else 0.0
    console.print(f" results {overall_pct}")
    overall_label = confidence_label(overall_pct)
    overall_color = confidence_color(overall_pct)

    # ── Send sources + per-source confidence ──────────────────────────────────
    seen    = set()
    sources = []
    for r in results:
        doc  = r["doc"]
        src  = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        k    = f"{src}-{page}"
        if k not in seen:
            seen.add(k)
            sources.append({
                "file":       src,
                "page":       page,
                "preview":    doc.page_content[:100].replace("\n"," "),
                "confidence": r["confidence"],
                "label":      r["label"],
                "color":      confidence_color(r["confidence"]),
            })

    yield sse({"type":"sources","data":sources})
    await asyncio.sleep(0)

    # ── Send overall confidence immediately ───────────────────────────────────
    yield sse({
        "type":  "confidence",
        "score": overall_pct,
        "label": overall_label,
        "color": overall_color,
    })
    await asyncio.sleep(0)

    # ── Build prompt ──────────────────────────────────────────────────────────
    context_parts = []
    total_len     = 0
    for doc in top_docs:
        if total_len + len(doc.page_content) > req.max_context: break
        context_parts.append(doc.page_content)
        total_len += len(doc.page_content)

    context  = "\n\n---\n\n".join(context_parts)

    # CONCEPT GRAPH HOOK: append structured concept knowledge (definitions,
    # formulas, known misconceptions, related concepts) alongside the raw
    # Chroma chunks above. get_context_for_topics() returns "" if the graph
    # is disabled, empty, or nothing matched -- context is then identical
    # to what this endpoint produced before this hook existed.
    graph_matches = 0
    if concept_graph.CONCEPT_GRAPH_ENABLED:
        graph_topics = req.topics or [req.question]
        graph_block  = concept_graph.get_context_for_topics(graph_topics, req.class_name, req.subject)
        graph_matches = concept_graph.match_count(graph_topics, req.class_name, req.subject)
        if graph_block:
            context = (
                f"{context}\n\n--- CONCEPT GRAPH KNOWLEDGE "
                f"(verified relationships & known student misconceptions) ---\n{graph_block}"
            )

    yield sse({"type":"graph","matched_concepts":graph_matches})
    await asyncio.sleep(0)

    lang_ins = LANGUAGE_INSTRUCTIONS.get(req.subject, "Respond in English.")
    mode_ins = ANSWER_MODE_INSTRUCTIONS.get(
        req.answer_mode, ANSWER_MODE_INSTRUCTIONS["detailed"]
    )

    # ── Grounded vs. general-knowledge fallback ────────────────────────────
    # Below the confidence floor, don't force the strict context-only prompt
    # onto weak/irrelevant chunks -- let the LLM answer from its own
    # CBSE-syllabus knowledge instead, clearly flagged so the frontend can
    # show it as such. See MIN_CONFIDENCE_FOR_GROUNDED_ANSWER above for why.
    context_sufficient = overall_pct >= MIN_CONFIDENCE_FOR_GROUNDED_ANSWER

    if context_sufficient:
        prompt = BASE_PROMPT.format(
            context=context,
            question=req.question,
            class_name=req.class_name.replace("_"," ").title(),
            subject=req.subject.replace("_"," ").title(),
            language_instruction=lang_ins,
            answer_mode_instruction=mode_ins,
        )
    else:
        prompt = GENERAL_KNOWLEDGE_PROMPT.format(
            question=req.question,
            class_name=req.class_name.replace("_"," ").title(),
            subject=req.subject.replace("_"," ").title(),
            language_instruction=lang_ins,
            answer_mode_instruction=mode_ins,
        )
        yield sse({
            "type": "notice",
            "message": "No closely matching textbook passage found — answering from general subject knowledge instead.",
        })
        await asyncio.sleep(0)

    yield sse({"type":"status","message":"Generating answer…"})
    await asyncio.sleep(0)

    # ── Stream tokens ─────────────────────────────────────────────────────────
    think_filter = ThinkTagFilter()
    any_visible  = False
    try:
        from langchain_core.messages import HumanMessage
        # NOTE: previously this sent req.question directly, bypassing the
        # `prompt` built above entirely -- meaning the retrieved textbook
        # context and anti-hallucination rules were never actually reaching
        # the model here. Fixed to use `prompt`.
        conversation = [HumanMessage(content=prompt)]
        async for token in _iter_with_timeout(
            get_rag_chain().astream_with_tools(conversation),
            timeout=LLM_STREAM_TIMEOUT_SECONDS,
        ):
            visible = think_filter.feed(token)
            if visible:
                any_visible = True
                yield sse({"type":"token","data":visible})
                await asyncio.sleep(0)
        trailing = think_filter.flush()
        if trailing:
            any_visible = True
            yield sse({"type":"token","data":trailing})
            await asyncio.sleep(0)

        # If literally every token was consumed by hidden <think> reasoning
        # (model ran long on reasoning and never got to visible output, or
        # cut off mid-thought), don't leave the user with a blank card --
        # force one more plain-text, no-tools turn asking directly for the
        # answer. This is a distinct case from the max_tool_iterations
        # forced-turn already inside astream_with_tools(): that one fires
        # when tool_calls are still pending at the iteration cap, this one
        # fires when the model produced tokens but none of them were ever
        # visible to the user.
        if not any_visible:
            conversation.append(HumanMessage(
                content="Answer the question directly now, in plain text, "
                        "with no <think> reasoning block and no further "
                        "tool calls -- just the final worked answer."
            ))
            async for token in _iter_with_timeout(
                get_rag_chain().astream_with_tools(conversation),
                timeout=LLM_STREAM_TIMEOUT_SECONDS,
            ):
                visible = think_filter.feed(token)
                if visible:
                    yield sse({"type":"token","data":visible})
                    await asyncio.sleep(0)
            trailing = think_filter.flush()
            if trailing:
                yield sse({"type":"token","data":trailing})
                await asyncio.sleep(0)
    except asyncio.TimeoutError:
        yield sse({
            "type":"error",
            "message":f"Generation stalled and didn't respond for {int(LLM_STREAM_TIMEOUT_SECONDS)}s. Please try again."
        })
        return
    except Exception as e:
        console.print(f"error {e}")
        yield sse({"type":"error","message":f"LLM error: {e}"}); return

    # ── Send images ───────────────────────────────────────────────────────────
    try:
        images = find_relevant_images(req.class_name, req.subject, top_docs, 3)
        if images:
            yield sse({"type":"images","data":[{
                "filename": img["filename"],
                "page":     img["page"],
                "source":   img["source_pdf"],
                "url":      f"/api/images/{img['filename']}"
            } for img in images]})
    except Exception:
        pass

    yield sse({"type":"done","message":"Answer complete","answer_mode":req.answer_mode})

# ── Worksheet SSE generator ───────────────────────────────────────────────────

# Explicit Bloom's taxonomy guidance per difficulty so the LLM never
# defaults to recall-level questions regardless of what difficulty is set.
DIFFICULTY_GUIDANCE = {
    "easy": """DIFFICULTY — EASY (Bloom's Level 1-2: Remember & Understand)
- Ask students to recall facts, define terms, identify, label, or list.
- MCQ distractors should be clearly wrong — not tricky.
- Short answers need only 1-2 sentences.
- Fill-in-the-blank should have obvious answers directly from the text.
- Assertion-Reason: both statements should be independently verifiable
  facts straight from the text; avoid testing whether R "explains" A.
- Avoid inference, calculation, or multi-step reasoning entirely.
- Verbs to use: define, list, state, name, identify, recall, label.""",

    "medium": """DIFFICULTY — MEDIUM (Bloom's Level 3-4: Apply & Analyse)
- Ask students to explain why/how, compare, classify, or solve.
- MCQ distractors must be plausible — students need real understanding to rule them out.
- Short answers need 3-5 sentences with explanation or an example.
- Include at least one calculation or step-by-step problem where subject allows.
- Assertion-Reason: both statements true, but mix in cases where R does
  NOT correctly explain A even though both are individually true.
- Verbs to use: explain, compare, classify, solve, demonstrate, differentiate, calculate.""",

    "hard": """DIFFICULTY — HARD (Bloom's Level 5-6: Evaluate & Create)
- Ask students to evaluate arguments, justify decisions, predict outcomes, or design solutions.
- MCQ should have 2-3 highly plausible distractors requiring deep reasoning to eliminate.
- Short/long answers must require multi-step reasoning, inference from data, or real-world application.
- Include scenario-based or case-study style questions.
- Require linking concepts across sections or chapters where possible.
- Assertion-Reason: use all four outcomes (A/B/C/D) across the set, and
  favor cases where spotting a FALSE reason or a non-explanatory true
  reason requires genuine conceptual understanding, not just recall.
- Verbs to use: evaluate, justify, predict, design, critique, infer, hypothesize, analyse.""",

    "mixed": """DIFFICULTY — MIXED (all Bloom's levels)
- Distribute: ~30% easy (recall), ~40% medium (apply/analyse), ~30% hard (evaluate/create).
- Do NOT make all questions the same difficulty — genuine variety across levels is mandatory.
- Progress from simpler to more challenging within each section where possible.""",
}

WORKSHEET_PROMPT = """You are an expert CBSE question paper setter for {class_name}, subject: {subject}.

{language_instruction}

Use ONLY the context below from NCERT textbooks to create worksheet questions.
If context is insufficient, draw from closely related NCERT concepts on the same topic.

TASK: Generate exactly {num_questions} worksheet questions on the topic(s): {topics}
Question types to include (distribute evenly): {question_types}
{extra}

{difficulty_guidance}

ANSWER KEY REQUIREMENT — every question you write must also carry its own
answer, generated right now while you still have the context in front of
you (this is far more reliable than answering it later from a fresh,
lower-confidence retrieval on the question text alone):
- For "mcq": "answer" = just the correct option letter (e.g. "B"),
  "explanation" = 1-2 sentences grounded in the CONTEXT for why it's correct.
- For "truefalse": "answer" = "TRUE" or "FALSE",
  "explanation" = 1-2 sentences grounded in the CONTEXT.
- For "fillblank": "answer" = the exact word/phrase for each blank, in
  order, comma-separated if there are multiple blanks.
- For "assertreason": write "assertion" (statement A) and "reason"
  (statement R) as SEPARATE fields, both grounded in the CONTEXT. Do NOT
  put them in "text". "answer" = just the correct option letter (A-D),
  chosen from the FIXED four options below (do not invent your own options
  or reword them):
    A. Both A and R are true and R is the correct explanation of A.
    B. Both A and R are true but R is NOT the correct explanation of A.
    C. A is true but R is false.
    D. A is false but R is true.
  "explanation" = 1-2 sentences grounded in the CONTEXT justifying both
  the truth value of A, the truth value of R, and (if both true) whether R
  actually explains A -- this is the part students get wrong, so be explicit.
  Good assertion-reason pairs test whether R is a *correct* explanation of
  A even when both statements are independently true (i.e. favor option B
  some of the time, not only A/C/D) -- don't default to always-A.
- For "short" / "long": "answer" = a model answer grounded in the CONTEXT,
  as many sentences/steps as the question warrants.
- If the question involves any arithmetic, algebra, or numeric result,
  ALWAYS call the `calculate` tool to get the exact value for the answer
  field instead of computing it mentally -- even if the underlying formula
  is from the CONTEXT and the specific numbers in the question are new.
  Applying a textbook formula to new numbers is normal question-setting,
  not hallucination.

STRICT JSON OUTPUT — return ONLY this JSON object, no markdown fences, no preamble:
{{
  "title": "Worksheet title",
  "subtitle": "Brief topic description",
  "instructions": "General student instructions (1-2 sentences)",
  "sections": [
    {{
      "sectionTitle": "Section name (e.g. Multiple Choice Questions)",
      "questions": [
        {{
          "type": "mcq",
          "text": "Question text here",
          "options": ["A. option1", "B. option2", "C. option3", "D. option4"],
          "answer": "B",
          "explanation": "Why B is correct, grounded in the context."
        }},
        {{
          "type": "short",
          "text": "Short answer question here",
          "answer": "The model answer, grounded in the context."
        }},
        {{
          "type": "truefalse",
          "text": "Statement for true/false",
          "answer": "TRUE",
          "explanation": "Why, grounded in the context."
        }},
        {{
          "type": "fillblank",
          "text": "Sentence with ___ for blank(s)",
          "answer": "the missing word or phrase"
        }},
        {{
          "type": "long",
          "text": "Essay or long answer question",
          "answer": "The full model answer, grounded in the context."
        }},
        {{
          "type": "assertreason",
          "assertion": "Statement A here",
          "reason": "Statement R here",
          "answer": "B",
          "explanation": "Why A is true/false, why R is true/false, and whether R explains A."
        }}
      ]
    }}
  ]
}}

HARD RULES — violating any makes output invalid:
- Group questions by type into separate sections
- EVERY question object MUST include a non-empty "answer" field as described above
- MCQ must have exactly 4 options labeled A–D; exactly one must be correct
- Assertion-Reason questions use "assertion"/"reason" fields, NEVER "text"
  or "options" -- the four options are fixed and rendered by the frontend,
  not written by you
- Fill-in-the-blank must place ___ in the sentence for each blank
- Questions must be grounded in the CONTEXT below
- SILENT TOOL USE: your entire visible output, from the very first character
  to the very last, must be the JSON object and NOTHING else. If you need
  the `calculate` tool for a numeric answer, call it with ZERO other text in
  that turn — no lead-in sentence like "Now let me compute...", no
  commentary, no explanation of what you're about to calculate. Call the
  tool silently, wait for its result, then resume writing the JSON exactly
  where you left off. Writing ANY narration around a tool call breaks the
  JSON and makes the entire worksheet fail to parse.
- Age-appropriate for {class_name}
- STRICTLY follow the difficulty guidance above — never default to easy recall questions
- NO DUPLICATE QUESTIONS: every question must test a DIFFERENT fact, concept, or skill.
  Do not ask the same thing twice even with different wording.
  Before writing each question, check it does not overlap with any previous question.
- NO REPETITION OF STEM: question stems must not start with the same phrase
- SPREAD ACROSS CONTEXT: draw questions from DIFFERENT sections of the context —
  do not pull all questions from the same paragraph or section heading
- LANGUAGE SUBJECT NOTE: The context is tagged with [lang_sub_type] and [sentence_class].
  Use these tags to create appropriate question formats:
    • [grammar_rule] + [grammar_example] → grammar MCQ, fill-in-blank, rewrite questions
    • [grammar_exercise] → use the exercise format directly (fill blanks, rewrite, match)
    • [comprehension] + [comprehension_question] → passage-based questions with sub-questions
    • [short_answer_q] + [short_sentence] → 1-2 sentence answer questions
    • [long_answer_q] + [passage] → paragraph / essay questions
    • [vocabulary] + [word_list] → synonym/antonym/match-the-word MCQ or fill-blank
    • [dialogue] → complete-the-dialogue or conversation-based questions
    • [letter_writing] / [essay_writing] / [report_writing] / [speech] → writing prompts
    • [poem] → comprehension, appreciation, or stanza-based questions
    • [story] → character, theme, sequence questions
    • [translation] → translate-the-sentence questions

---
CONTEXT:
{context}
"""

QTYPE_SECTION_NAMES = {
    "mcq":       "Multiple Choice Questions",
    "short":     "Short Answer Questions",
    "truefalse": "True or False",
    "fillblank": "Fill in the Blanks",
    "long":      "Long Answer / Essay Questions",
    "assertreason": "Assertion-Reason Questions",
}


def _randomize_mcq_options(worksheet: dict) -> dict:
    """Shuffle MCQ option order so the correct answer is not always at B or C.

    Rewrites each option's leading letter (A–D) after shuffling and updates
    the answer key to match the new position of the correct option.
    """
    for section in worksheet.get("sections", []):
        for q in section.get("questions", []):
            if q.get("type") != "mcq":
                continue
            options = q.get("options")
            if not isinstance(options, list) or len(options) < 2:
                continue
            answer_letter = str(q.get("answer", "")).upper().strip()
            parsed = []
            correct_idx = None
            for i, opt in enumerate(options):
                m = re.match(r"^([A-D])\.\s*(.*)$", opt.strip(), re.IGNORECASE)
                if m:
                    letter = m.group(1).upper()
                    text = m.group(2)
                    if letter == answer_letter:
                        correct_idx = i
                    parsed.append(text)
                else:
                    parsed.append(opt)
            if correct_idx is None:
                continue
            indices = list(range(len(parsed)))
            random.shuffle(indices)
            new_options = [f"{chr(ord('A') + i)}. {parsed[j]}" for i, j in enumerate(indices)]
            new_answer = chr(ord('A') + indices.index(correct_idx))
            q["options"] = new_options
            q["answer"] = new_answer
    return worksheet


async def worksheet_stream(req: WorksheetRequest) -> AsyncGenerator[str, None]:
    if req.class_name not in SUPPORTED_CLASSES:
        yield sse({"type":"error","message":f"Invalid class: {req.class_name}"}); return
    if req.subject not in SUPPORTED_SUBJECTS:
        yield sse({"type":"error","message":f"Invalid subject: {req.subject}"}); return
    if not req.question_types:
        yield sse({"type":"error","message":"Select at least one question type."}); return

    yield sse({"type":"status","message":"Searching NCERT textbooks for relevant content…"})
    await asyncio.sleep(0)

    # ── Bloom level target for this difficulty ────────────────────────────────
    BLOOM_FOR_DIFFICULTY = {
        "easy":   {"remember", "understand"},
        "medium": {"understand", "apply", "analyse"},
        "hard":   {"analyse", "evaluate"},
        "mixed":  {"remember", "understand", "apply", "analyse", "evaluate"},
    }
    # Content types most useful for worksheet generation (ordered by priority)
    PREFERRED_TYPES = [
        "question", "answer", "example", "definition",
        "fact", "formula", "summary", "body",
        "exercise", "note", "activity", "table",
        "figure_ref", "introduction",
    ]
    # For language subjects — which lang_sub_types are richest for worksheet generation
    LANG_PREFERRED_TYPES = [
        "grammar_exercise", "comprehension_question", "short_answer_q",
        "long_answer_q", "comprehension", "grammar_rule", "grammar_example",
        "vocabulary", "dialogue", "letter_writing", "essay_writing",
        "story", "poem", "summary_passage", "translation",
        "note_making", "report_writing", "speech", "body",
    ]
    # sentence_class preferences by question_type requested
    SENTENCE_CLASS_FOR_QTYPE = {
        "short":     {"short_sentence", "long_sentence"},
        "fillblank": {"short_sentence", "word_list"},
        "long":      {"passage", "long_sentence"},
        "mcq":       {"short_sentence", "long_sentence", "passage"},
        "truefalse": {"short_sentence", "long_sentence"},
        "assertreason": {"short_sentence", "long_sentence"},
    }

    _LANG_SUBJECTS = {"english", "hindi", "kannada", "tamil", "sanskrit"}
    is_lang_subject = req.subject in _LANG_SUBJECTS
    target_blooms   = BLOOM_FOR_DIFFICULTY.get(req.difficulty, BLOOM_FOR_DIFFICULTY["mixed"])

    # ── Retrieve for each topic ───────────────────────────────────────────────
    try:
        retriever = get_rag_chain().retriever
        all_results = []
        seen_keys        = set()   # dedup by source+page
        seen_fingerprints = set()  # dedup by content similarity (first 120 chars)

        for topic in req.topics:
            results = retriever.retrieve_and_rerank(
                topic,
                class_filter=req.class_name,
                subject_filter=req.subject
            )
            for r in results:
                doc = r["doc"]
                key = f"{doc.metadata.get('source','')}:{doc.metadata.get('page','')}"
                # Content fingerprint — skip chunks whose first 120 chars match
                # (catches the same passage split across two chunks)
                fp = doc.page_content[:120].strip().lower()
                if key not in seen_keys and fp not in seen_fingerprints:
                    seen_keys.add(key)
                    seen_fingerprints.add(fp)
                    all_results.append(r)
    except Exception as e:
        yield sse({"type":"error","message":f"Retrieval error: {e}"}); return

    if not all_results:
        yield sse({"type":"error","message":"No relevant content found. Make sure PDFs are ingested for this class/subject."}); return

    # ── Re-rank results by metadata quality for worksheet use ─────────────────
    def _ws_score(r: dict) -> float:
        doc  = r["doc"]
        meta = doc.metadata
        base = r["confidence"]

        # Bloom level match boost
        bloom = meta.get("bloom_level", "remember")
        bloom_boost = 15.0 if bloom in target_blooms else 0.0

        # Strongly deprioritise in-text questions (not exam-appropriate)
        loc = meta.get("question_location", "body")
        loc_penalty = -20.0 if loc == "intext" else (5.0 if loc == "exercise" else 0.0)

        # Penalise chunks that require a diagram (can't answer in text worksheet)
        diagram_penalty = -10.0 if meta.get("requires_diagram") else 0.0

        if is_lang_subject:
            lst = meta.get("lang_sub_type", "body")
            try:    type_rank = LANG_PREFERRED_TYPES.index(lst)
            except: type_rank = len(LANG_PREFERRED_TYPES)
            type_boost = max(0, (len(LANG_PREFERRED_TYPES) - type_rank) * 2.0)

            sc = meta.get("sentence_class", "")
            sc_boost = 0.0
            for qt in req.question_types:
                if sc in SENTENCE_CLASS_FOR_QTYPE.get(qt, set()):
                    sc_boost = max(sc_boost, 8.0)
            formula_boost = 0.0
        else:
            ctype = meta.get("content_type", "body")
            try:    type_rank = PREFERRED_TYPES.index(ctype)
            except: type_rank = len(PREFERRED_TYPES)
            type_boost = max(0, (len(PREFERRED_TYPES) - type_rank) * 1.5)
            sc_boost   = 0.0
            formula_boost = 3.0 if meta.get("has_formula") and req.subject in (
                "mathematics", "science", "physics", "chemistry"
            ) else 0.0

        return base + bloom_boost + type_boost + sc_boost + formula_boost + loc_penalty + diagram_penalty

    all_results.sort(key=_ws_score, reverse=True)

    # ── Overall confidence ────────────────────────────────────────────────────
    weights  = [1.0, 0.8, 0.6, 0.4, 0.2]
    total_w  = sum(weights[i] if i < len(weights) else 0.1 for i in range(len(all_results)))
    total_s  = sum(
        all_results[i]["confidence"] * (weights[i] if i < len(weights) else 0.1)
        for i in range(len(all_results))
    )
    overall_pct   = round(total_s / total_w, 1) if total_w else 0.0
    overall_label = confidence_label(overall_pct)
    overall_color = confidence_color(overall_pct)

    yield sse({
        "type":"confidence","score":overall_pct,
        "label":overall_label,"color":overall_color,
    })
    await asyncio.sleep(0)

    # ── Send enriched sources to UI ───────────────────────────────────────────
    seen    = set()
    sources = []
    for r in all_results:
        doc  = r["doc"]
        meta = doc.metadata
        src  = meta.get("source","Unknown")
        page = meta.get("page","?")
        k    = f"{src}-{page}"
        if k not in seen:
            seen.add(k)
            sources.append({
                "file":         os.path.basename(src),
                "page":         page,
                "chapter":      meta.get("chapter",""),
                "section":      meta.get("section",""),
                "content_type": meta.get("content_type","body"),
                "bloom_level":  meta.get("bloom_level",""),
                "preview":      doc.page_content[:120].replace("\n"," "),
                "confidence":   r["confidence"],
                "label":        r["label"],
                "color":        confidence_color(r["confidence"]),
            })
    yield sse({"type":"sources","data":sources})
    await asyncio.sleep(0)

    # ── Build context — ensure section diversity to prevent duplicate Qs ─────
    top_docs = [r["doc"] for r in all_results]
    context_parts, total_len = [], 0
    seen_sections = {}   # section → count; cap at 2 chunks per section

    for doc in top_docs:
        if total_len + len(doc.page_content) > req.max_context:
            break
        meta    = doc.metadata
        section = meta.get("section") or meta.get("chapter") or "root"

        # Allow at most 2 chunks from the same section to force topic spread
        if seen_sections.get(section, 0) >= 2:
            continue
        seen_sections[section] = seen_sections.get(section, 0) + 1

        header_parts = []
        if meta.get("chapter"):       header_parts.append(meta["chapter"])
        if meta.get("section"):       header_parts.append(meta["section"])
        if meta.get("subsection"):    header_parts.append(meta["subsection"])
        if meta.get("content_type"):  header_parts.append(f"[{meta['content_type']}]")
        if meta.get("science_domain"):header_parts.append(f"[{meta['science_domain']}]")
        if meta.get("social_domain"): header_parts.append(f"[{meta['social_domain']}]")
        if meta.get("lang_sub_type"): header_parts.append(f"[{meta['lang_sub_type']}]")
        if meta.get("sentence_class"):header_parts.append(f"[{meta['sentence_class']}]")
        if meta.get("bloom_level"):   header_parts.append(f"[bloom:{meta['bloom_level']}]")
        if meta.get("question_location") == "exercise": header_parts.append("[exercise_question]")

        header = " | ".join(header_parts)
        part   = (f"{header}\n{doc.page_content}") if header else doc.page_content
        context_parts.append(part)
        total_len += len(doc.page_content)
        header = " | ".join(header_parts)
        part   = (f"{header}\n{doc.page_content}") if header else doc.page_content
        context_parts.append(part)
        total_len += len(doc.page_content)
    context = "\n\n---\n\n".join(context_parts)

    # CONCEPT GRAPH HOOK: same append-only pattern as answer_stream. Worksheet
    # generation is the primary beneficiary -- misconceptions/contrasts here
    # are what let the LLM write genuinely tricky (not just recall) questions
    # instead of relying on it to invent plausible-sounding distractors cold.
    graph_matches = 0
    if concept_graph.CONCEPT_GRAPH_ENABLED:
        graph_block   = concept_graph.get_context_for_topics(req.topics, req.class_name, req.subject)
        graph_matches = concept_graph.match_count(req.topics, req.class_name, req.subject)
        if graph_block:
            context = (
                f"{context}\n\n--- CONCEPT GRAPH KNOWLEDGE "
                f"(verified relationships & known student misconceptions -- use these "
                f"to sharpen distractors and case-study twists) ---\n{graph_block}"
            )

    yield sse({"type":"graph","matched_concepts":graph_matches})
    await asyncio.sleep(0)

    lang_ins = LANGUAGE_INSTRUCTIONS.get(req.subject, "Respond in English.")

    qtype_labels = {
        "mcq":"Multiple Choice","short":"Short Answer",
        "truefalse":"True/False","fillblank":"Fill in the Blank","long":"Long Answer/Essay",
        "assertreason":"Assertion-Reason"
    }

    prompt = WORKSHEET_PROMPT.format(
        class_name=req.class_name.replace("_"," ").title(),
        subject=req.subject.replace("_"," ").title(),
        language_instruction=lang_ins,
        num_questions=req.num_questions,
        topics=", ".join(req.topics),
        difficulty_guidance=DIFFICULTY_GUIDANCE.get(req.difficulty, DIFFICULTY_GUIDANCE["medium"]),
        question_types=", ".join(qtype_labels.get(t, t) for t in req.question_types),
        extra=f"Extra instructions: {req.extra_instructions}" if req.extra_instructions else "",
        context=context,
    )

    yield sse({"type":"status","message":"Generating worksheet questions from textbook content…"})
    await asyncio.sleep(0)

    # Stream LLM output and collect full response
    full_response = ""
    think_filter  = ThinkTagFilter()
    try:
        from langchain_core.messages import HumanMessage
        async for token in _iter_with_timeout(
            get_rag_chain().astream_with_tools([HumanMessage(content=prompt)]),
            timeout=LLM_STREAM_TIMEOUT_SECONDS,
        ):
            visible = think_filter.feed(token)
            if visible:
                full_response += visible
                yield sse({"type":"progress","data":visible})
                await asyncio.sleep(0)
        trailing = think_filter.flush()
        if trailing:
            full_response += trailing
            yield sse({"type":"progress","data":trailing})
            await asyncio.sleep(0)
    except asyncio.TimeoutError:
        yield sse({
            "type":"error",
            "message":f"Generation stalled (likely during a calculation step) and " \
                       f"didn't respond for {int(LLM_STREAM_TIMEOUT_SECONDS)}s. Please try again — if it keeps " \
                       f"happening, try fewer questions or a narrower topic."
        })
        return
    except Exception as e:
        yield sse({"type":"error","message":f"LLM error: {e}"}); return

    # Parse JSON from LLM output
    try:
        clean = re.sub(r"^```(?:json)?", "", full_response.strip()).strip()
        clean = re.sub(r"```$", "", clean).strip()
        # Find JSON object in response
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)
        worksheet = json.loads(clean)
        _randomize_mcq_options(worksheet)
        yield sse({"type":"worksheet","data":worksheet})
    except Exception as e:
        yield sse({"type":"error","message":f"Could not parse worksheet JSON: {e}. Try again."}); return

    yield sse({"type":"done","message":"Worksheet generated successfully"})


@app.post("/api/generate-worksheet")
async def generate_worksheet(req: WorksheetRequest):
    return StreamingResponse(
        worksheet_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ── Page routes ───────────────────────────────────────────────────────────────
@app.get("/",            response_class=HTMLResponse)
async def root():             return FileResponse(STATIC_DIR / "index.html")

@app.get("/upload",      response_class=HTMLResponse)
async def upload_page():      return FileResponse(STATIC_DIR / "upload.html")

@app.get("/quiz",        response_class=HTMLResponse)
async def quiz_page():        return FileResponse(STATIC_DIR / "quiz.html")

@app.get("/worksheet",   response_class=HTMLResponse)
async def worksheet_page():   return FileResponse(STATIC_DIR / "worksheet.html")

# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":         "ok",
        "chroma_db":      os.path.exists(CHROMA_DIR),
        "data_dir":       os.path.exists(DATA_DIR),
        "llm_provider":   os.getenv("LLM_PROVIDER","deepseek"),
        "embed_provider": os.getenv("EMBED_PROVIDER","ollama"),
    }

@app.get("/api/classes")
def list_classes():
    if not os.path.exists(DATA_DIR): return {"classes":[]}
    return {"classes":[
        {"id":d,"label":d.replace("_"," ").title()}
        for d in sorted(os.listdir(DATA_DIR))
        if os.path.isdir(os.path.join(DATA_DIR,d)) and d in SUPPORTED_CLASSES
    ]}

@app.get("/api/subjects/{class_name}")
def list_subjects(class_name: str):
    if class_name not in SUPPORTED_CLASSES:
        raise HTTPException(400, f"Invalid class: {class_name}")
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        raise HTTPException(404, f"No folder for {class_name}")
    tracker = load_tracker()
    return {"class":class_name,"subjects":[{
        "id":d,"label":d.replace("_"," ").title(),
        "ingested":sum(1 for k in tracker if k.startswith(f"{class_name}/{d}/"))
    } for d in sorted(os.listdir(class_path))
      if os.path.isdir(os.path.join(class_path,d))]}

@app.get("/api/files/{class_name}/{subject}")
def list_files(class_name: str, subject: str):
    subject_path = os.path.join(DATA_DIR, class_name, subject)
    if not os.path.exists(subject_path):
        raise HTTPException(404, f"No folder: {class_name}/{subject}")
    tracker = load_tracker()
    return {"class":class_name,"subject":subject,"files":[{
        "filename":  fname,
        "ingested":  f"{class_name}/{subject}/{fname}" in tracker,
        "chunks":    tracker.get(f"{class_name}/{subject}/{fname}",{}).get("chunks",0),
        "ingested_at":tracker.get(f"{class_name}/{subject}/{fname}",{}).get("ingested_at"),
    } for fname in sorted(os.listdir(subject_path)) if fname.endswith(".pdf")]}

@app.post("/api/ask")
async def ask_question(req: AskRequest):
    return StreamingResponse(
        answer_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.post("/api/upload")
async def upload_pdf(
    file:       UploadFile = File(...),
    class_name: str        = Form(...),
    subject:    str        = Form(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    if class_name not in SUPPORTED_CLASSES:
        raise HTTPException(400, f"Invalid class: {class_name}")
    if subject not in SUPPORTED_SUBJECTS:
        raise HTTPException(400, f"Invalid subject: {subject}")

    dest_dir  = os.path.join(DATA_DIR, class_name, subject)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, file.filename)

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        import hashlib
        from src.ingest import (
            get_embeddings, ingest_pdf_list,
            load_tracker as _load_tracker, save_tracker
        )
        from langchain_community.vectorstores import Chroma

        hasher = hashlib.md5()
        with open(dest_path,"rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""): hasher.update(chunk)

        pdf_info = {
            "path":     dest_path,
            "class":    class_name,
            "subject":  subject,
            "filename": file.filename,
            "key":      f"{class_name}/{subject}/{file.filename}",
            "hash":     hasher.hexdigest()
        }

        embeddings  = get_embeddings()
        tracker     = _load_tracker()
        vectorstore = None
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )

        batch_size    = int(os.getenv("EMBED_BATCH_SIZE", 10))
        chunk_size    = int(os.getenv("CHUNK_SIZE", 600))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 80))

        vectorstore, total_failed = ingest_pdf_list(
            [pdf_info], embeddings, vectorstore,
            CHROMA_DIR, chunk_size, chunk_overlap,
            batch_size, tracker, force=True
        )
        chunks = tracker.get(pdf_info["key"],{}).get("chunks", 0)

        # CONCEPT GRAPH HOOK (2/2): build/update the concept graph for this
        # chapter. Wrapped so a graph-extraction failure never blocks the
        # ingestion response the UI is waiting on -- Chroma ingestion above
        # has already succeeded and stands on its own either way.
        graph_result = {"status": "skipped"}
        if concept_graph.CONCEPT_GRAPH_ENABLED:
            try:
                chapter_text = _extract_pdf_text(dest_path)
                graph_result = concept_graph.build_from_chapter_text(
                    chapter_text=chapter_text,
                    class_name=class_name,
                    subject=subject,
                    chapter=file.filename,
                    llm_raw=get_rag_chain().llm_raw,
                    source=file.filename,
                )
            except Exception as e:
                console.print(f"[red]concept_graph build failed for {file.filename}: {e}[/red]")
                graph_result = {"status": "error", "message": str(e)}

        return {
            "status": "ok", "filename": file.filename, "chunks": chunks,
            "concept_graph": graph_result,
        }

    except Exception as e:
        raise HTTPException(500, f"Ingestion error: {str(e)}")

@app.post("/api/extract-questions")
async def extract_questions(
    file:       UploadFile = File(...),
    class_name: str        = Form(...),
    subject:    str        = Form(...)
):
    filename_lower = file.filename.lower()
    if not filename_lower.endswith((".pdf", ".html", ".htm")):
        raise HTTPException(400, "Only PDF or HTML files accepted")

    # ── HTML path: structured extraction, no LLM call needed ───────────────
    # Handles Mathpix-style OCR exports (photo -> HTML) where each formula's
    # exact LaTeX already ships as a hidden sibling next to its rendered SVG
    # glyphs -- see src/html_question_extractor.py for why this is reliable
    # without OCR/vision. Type classification is a free local heuristic
    # (guess_question_type), not an LLM call, since the questions are
    # already cleanly segmented by the extractor -- there's nothing for an
    # LLM re-extraction pass to add here, only cost/latency.
    if filename_lower.endswith((".html", ".htm")):
        try:
            raw_bytes = await file.read()
            extracted = extract_questions_from_html(raw_bytes)

            if not extracted:
                raise HTTPException(422, "Could not extract any questions from this HTML file")

            valid = [
                {"text": q.text, "type": guess_question_type(q.text)}
                for q in extracted
            ]
            return {"questions": valid, "total": len(valid)}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"HTML extraction error: {str(e)}")

    # ── PDF path: existing text extraction + LLM re-structuring ────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = _extract_pdf_text(tmp_path)
        print("extracted text length:", len(text))

        if not text.strip():
            raise HTTPException(422, "Could not extract text from PDF")

        extraction_prompt = f"""Extract ALL questions from this question paper.
For each question return JSON with "text" (exact question) and "type" (mcq/true_false/short/long/fill).
Return ONLY a JSON array, no markdown, no explanation:
[{{"text":"...","type":"short"}}]

PAPER:
{text[:6000]}"""

        from langchain_core.messages import HumanMessage
        response = await get_rag_chain().llm.ainvoke([HumanMessage(content=extraction_prompt)])
        print("LLM extraction response length:", len(response.content))
        if not response.content or not response.content.strip():
            # Distinguish "provider returned nothing" (rate-limit/error) from
            # a genuine parse failure -- otherwise this falls through to the
            # json.JSONDecodeError handler below and gets misreported as a
            # PDF-quality problem ("try a cleaner PDF") when the PDF was fine.
            raise RuntimeError(
                f"LLM provider returned an empty response during question "
                f"extraction. LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'unknown')!r}."
            )

        raw = re.sub(r"^```(?:json)?","",response.content.strip()).strip()
        raw = re.sub(r"```$","",raw).strip()

        questions = json.loads(raw)
        allowed   = {"mcq","true_false","short","long","fill"}
        valid     = [
            {"text":q["text"].strip(), "type":q.get("type","short") if q.get("type") in allowed else "short"}
            for q in questions if isinstance(q,dict) and q.get("text","").strip()
        ]
        return {"questions":valid,"total":len(valid)}

    except json.JSONDecodeError:
        print("JSON decode error during question extraction")
        raise HTTPException(422, "Could not parse questions — try a cleaner PDF")
    except Exception as e:
        print("Error during question extraction:", e)
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

@app.get("/api/images/{filename}")
def get_image(filename: str):
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    img_path = os.path.join(IMAGE_STORE_DIR, filename)
    if not os.path.exists(img_path):
        raise HTTPException(404, f"Image not found: {filename}")
    return FileResponse(
        img_path, media_type="image/png",
        headers={"Cache-Control":"public,max-age=86400"}
    )

@app.get("/api/images")
def list_images(
    class_name: Optional[str] = Query(None),
    subject:    Optional[str] = Query(None),
    page:       Optional[int] = Query(None)
):
    metadata   = load_image_metadata()
    all_images = metadata.get("images", [])
    filtered   = [
        {"filename":img["filename"],"class":img["class"],"subject":img["subject"],
         "source":img["source_pdf"],"page":img["page"],"url":f"/api/images/{img['filename']}"}
        for img in all_images
        if (not class_name or img["class"]==class_name)
        and (not subject    or img["subject"]==subject)
        and (page is None   or img["page"]==page)
    ]
    return {"total":len(filtered),"images":filtered}

@app.get("/api/ingest/status")
def ingest_status():
    tracker = load_tracker()
    return {"total_files":len(tracker),"files":[{
        "key":k,
        "class":info.get("class",k.split("/")[0]),
        "subject":info.get("subject",k.split("/")[1]),
        "filename":info.get("filename",k.split("/")[2]),
        "chunks":info.get("chunks",0),
        "ingested_at":info.get("ingested_at"),
        "images_extracted":info.get("images_extracted",False),
    } for k,info in sorted(tracker.items())]}
