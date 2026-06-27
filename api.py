"""
CBSE RAG - FastAPI Server
Endpoints:
  GET  /health                          → health check
  GET  /api/classes                     → list available classes
  GET  /api/subjects/{class_name}       → list subjects for a class
  GET  /api/files/{class_name}/{subject}→ list ingested PDFs
  POST /api/ask                         → SSE streaming answer
  GET  /api/images/{filename}           → serve image file
  GET  /api/ingest/status               → ingestion tracker status
"""

import os
import json
import asyncio
import re
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="CBSE RAG API",
    description="CBSE Class 8-10 RAG system with SSE streaming",
    version="1.0.0"
)

# Allow all origins for development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    "hindi":        "हिंदी में जवाब दें।",
    "kannada":      "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
    "tamil":        "தமிழில் பதில் சொல்லுங்கள்.",
    "sanskrit":     "संस्कृते उत्तरं देहि।",
    "mathematics":  "Respond in English with clear numbered step-by-step solutions.",
    "science":      "Respond in English with clear explanations.",
    "socialscience": "Respond in English.",
    "english":      "Respond in English.",
}

MATH_PROMPT = """You are an expert CBSE tutor for {class_name}, subject: {subject}.

{language_instruction}

Use ONLY the context below from NCERT textbooks to answer.
If the answer is not in the context, say so clearly.

Rules:
- For maths: show every step clearly, numbered (Step 1, Step 2...)
- Define formulas before using them
- Use simple language for a school student
- End with a short summary or key takeaway
- Use markdown formatting (bold for key terms, use ``` for formulas)
- When referencing a diagram or figure mention its filename if available

---
CONTEXT:
{context}

---
QUESTION: {question}

ANSWER:"""

# ── Lazy singletons ───────────────────────────────────────────────────────────
# Load retriever and LLM once at startup, reuse across requests

_retriever = None
_llm       = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retriever import CBSERetriever
        _retriever = CBSERetriever()
    return _retriever

def get_llm():
    global _llm
    if _llm is None:
        provider = os.getenv("LLM_PROVIDER", "groq")
        if provider == "groq":
            from langchain_groq import ChatGroq
            _llm = ChatGroq(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1,
                max_tokens=2048,
            )
        elif provider == "deepseek":
            from langchain_openai import ChatOpenAI
            _llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                temperature=0.1,
                max_tokens=2048,
            )
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm = ChatGoogleGenerativeAI(
                model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.1,
            )
    return _llm

# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:    str              = Field(..., min_length=3, example="What is the quadratic formula?")
    class_name:  str              = Field(..., example="class_10")
    subject:     str              = Field(..., example="mathematics")
    max_context: int              = Field(2000, ge=500, le=5000)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tracker() -> dict:
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE) as f:
            return json.load(f)
    return {}

def load_image_metadata() -> dict:
    meta_file = os.path.join(IMAGE_STORE_DIR, "metadata.json")
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            return json.load(f)
    return {}

def find_relevant_images(
    class_name:  str,
    subject:     str,
    source_docs: list,
    max_images:  int = 3
) -> list:
    """Return image metadata dicts relevant to source doc pages."""
    metadata   = load_image_metadata()
    all_images = metadata.get("images", [])
    if not all_images:
        return []

    referenced = [
        {"source": d.metadata.get("source", ""), "page": d.metadata.get("page", -1)}
        for d in source_docs
        if d.metadata.get("source") and d.metadata.get("page", -1) >= 0
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
        if img.get("has_drawings"):
            score += 2
        if score > 0:
            scored.append((score, img))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [img for _, img in scored[:max_images]]

def clean_answer(text: str) -> str:
    """Strip DeepSeek <think> tags from output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def sse_event(data: dict) -> str:
    """Format a dict as an SSE event string."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# ── SSE Generator ─────────────────────────────────────────────────────────────

async def answer_stream(req: AskRequest) -> AsyncGenerator[str, None]:
    """
    Core SSE generator:
      1. Retrieve + re-rank docs
      2. Stream LLM tokens
      3. Send images event at the end
    """
    # Validate inputs
    if req.class_name not in SUPPORTED_CLASSES:
        yield sse_event({"type": "error", "message": f"Invalid class: {req.class_name}"})
        return

    if req.subject not in SUPPORTED_SUBJECTS:
        yield sse_event({"type": "error", "message": f"Invalid subject: {req.subject}"})
        return

    # ── Step 1: Retrieval ─────────────────────────────────────
    yield sse_event({"type": "status", "message": "Searching NCERT textbooks..."})
    await asyncio.sleep(0)  # yield control to event loop

    try:
        retriever  = get_retriever()
        top_docs   = retriever.retrieve_and_rerank(
            req.question,
            class_filter=req.class_name,
            subject_filter=req.subject
        )
    except Exception as e:
        yield sse_event({"type": "error", "message": f"Retrieval error: {str(e)}"})
        return

    if not top_docs:
        yield sse_event({
            "type":    "error",
            "message": "No relevant content found. Make sure PDFs are ingested for this class/subject."
        })
        return

    # Send source metadata to client immediately (before LLM starts)
    sources = []
    seen    = set()
    for doc in top_docs:
        src  = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page",   "?")
        key  = f"{src}-{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file":    src,
                "page":    page,
                "preview": doc.page_content[:100].replace("\n", " ")
            })

    yield sse_event({"type": "sources", "data": sources})
    await asyncio.sleep(0)

    # ── Step 2: Build prompt ──────────────────────────────────
    context_parts = []
    total_len     = 0
    for doc in top_docs:
        if total_len + len(doc.page_content) > req.max_context:
            break
        context_parts.append(doc.page_content)
        total_len += len(doc.page_content)

    context  = "\n\n---\n\n".join(context_parts)
    lang_ins = LANGUAGE_INSTRUCTIONS.get(req.subject, "Respond in English.")
    prompt   = MATH_PROMPT.format(
        context=context,
        question=req.question,
        class_name=req.class_name.replace("_", " ").title(),
        subject=req.subject.replace("_", " ").title(),
        language_instruction=lang_ins
    )

    # ── Step 3: Stream LLM tokens ─────────────────────────────
    yield sse_event({"type": "status", "message": "Generating answer..."})
    await asyncio.sleep(0)

    try:
        from langchain_core.messages import HumanMessage
        llm         = get_llm()
        full_answer = ""

        # Stream token by token
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            token = chunk.content
            if token:
                full_answer += token
                # Clean think tags from each streamed chunk
                visible = re.sub(r"<think>.*", "", token, flags=re.DOTALL)
                if visible:
                    yield sse_event({"type": "token", "data": visible})
                    await asyncio.sleep(0)

    except Exception as e:
        yield sse_event({"type": "error", "message": f"LLM error: {str(e)}"})
        return

    # ── Step 4: Send relevant images ──────────────────────────
    try:
        images = find_relevant_images(
            class_name=req.class_name,
            subject=req.subject,
            source_docs=top_docs,
            max_images=3
        )
        if images:
            image_data = [
                {
                    "filename": img["filename"],
                    "page":     img["page"],
                    "source":   img["source_pdf"],
                    "url":      f"/api/images/{img['filename']}"
                }
                for img in images
            ]
            yield sse_event({"type": "images", "data": image_data})
    except Exception:
        pass  # images are optional

    # ── Step 5: Done ──────────────────────────────────────────
    yield sse_event({"type": "done", "message": "Answer complete"})

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":     "ok",
        "chroma_db":  os.path.exists(CHROMA_DIR),
        "data_dir":   os.path.exists(DATA_DIR),
        "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
        "embed_provider": os.getenv("EMBED_PROVIDER", "ollama")
    }

@app.get("/api/classes")
def list_classes():
    """List all classes that have a folder in data/."""
    if not os.path.exists(DATA_DIR):
        return {"classes": []}
    available = [
        {
            "id":    d,
            "label": d.replace("_", " ").title()
        }
        for d in sorted(os.listdir(DATA_DIR))
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and d in SUPPORTED_CLASSES
    ]
    return {"classes": available}

@app.get("/api/subjects/{class_name}")
def list_subjects(class_name: str):
    """List subjects available for a class."""
    if class_name not in SUPPORTED_CLASSES:
        raise HTTPException(400, f"Invalid class: {class_name}")

    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        raise HTTPException(404, f"No data folder for {class_name}")

    tracker   = load_tracker()
    subjects  = []
    for d in sorted(os.listdir(class_path)):
        if not os.path.isdir(os.path.join(class_path, d)):
            continue
        # Count ingested files for this subject
        ingested = sum(
            1 for k in tracker
            if k.startswith(f"{class_name}/{d}/")
        )
        subjects.append({
            "id":       d,
            "label":    d.replace("_", " ").title(),
            "ingested": ingested
        })
    return {"class": class_name, "subjects": subjects}

@app.get("/api/files/{class_name}/{subject}")
def list_files(class_name: str, subject: str):
    """List PDF files for a class/subject with ingestion status."""
    if class_name not in SUPPORTED_CLASSES:
        raise HTTPException(400, f"Invalid class: {class_name}")

    subject_path = os.path.join(DATA_DIR, class_name, subject)
    if not os.path.exists(subject_path):
        raise HTTPException(404, f"No folder: {class_name}/{subject}")

    tracker = load_tracker()
    files   = []
    for fname in sorted(os.listdir(subject_path)):
        if not fname.endswith(".pdf"):
            continue
        key      = f"{class_name}/{subject}/{fname}"
        info     = tracker.get(key, {})
        files.append({
            "filename":         fname,
            "ingested":         key in tracker,
            "chunks":           info.get("chunks", 0),
            "ingested_at":      info.get("ingested_at", None),
            "images_extracted": info.get("images_extracted", False)
        })
    return {"class": class_name, "subject": subject, "files": files}

@app.post("/api/ask")
async def ask_question(req: AskRequest):
    """
    Stream answer as Server-Sent Events.

    SSE event types:
      status  → { type, message }          processing updates
      sources → { type, data: [...] }      source docs (sent before tokens)
      token   → { type, data: "..." }      one LLM token
      images  → { type, data: [...] }      relevant diagram images
      error   → { type, message }          error occurred
      done    → { type, message }          stream complete
    """
    return StreamingResponse(
        answer_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",    # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/api/images/{filename}")
def get_image(filename: str):
    """Serve an extracted diagram image by filename."""
    # Security: prevent path traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")

    img_path = os.path.join(IMAGE_STORE_DIR, filename)
    if not os.path.exists(img_path):
        raise HTTPException(404, f"Image not found: {filename}")

    return FileResponse(
        img_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"}
    )

@app.get("/api/images")
def list_images(
    class_name: str = Query(None),
    subject:    str = Query(None),
    page:       int = Query(None)
):
    """List available images, optionally filtered by class/subject/page."""
    metadata   = load_image_metadata()
    all_images = metadata.get("images", [])

    filtered = []
    for img in all_images:
        if class_name and img["class"]   != class_name: continue
        if subject    and img["subject"] != subject:     continue
        if page is not None and img["page"] != page:     continue
        filtered.append({
            "filename": img["filename"],
            "class":    img["class"],
            "subject":  img["subject"],
            "source":   img["source_pdf"],
            "page":     img["page"],
            "url":      f"/api/images/{img['filename']}"
        })

    return {"total": len(filtered), "images": filtered}

@app.get("/api/ingest/status")
def ingest_status():
    """Return ingestion tracker — what's been indexed."""
    tracker = load_tracker()
    summary = []
    for key, info in sorted(tracker.items()):
        summary.append({
            "key":              key,
            "class":            info.get("class",    key.split("/")[0]),
            "subject":          info.get("subject",  key.split("/")[1]),
            "filename":         info.get("filename", key.split("/")[2]),
            "chunks":           info.get("chunks",   0),
            "ingested_at":      info.get("ingested_at", None),
            "images_extracted": info.get("images_extracted", False)
        })
    return {
        "total_files": len(tracker),
        "files":       summary
    }
