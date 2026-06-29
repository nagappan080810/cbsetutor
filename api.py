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
  POST /api/extract-questions           → Extract questions from PDF
  GET  /api/images/{filename}
  GET  /api/images
  GET  /api/ingest/status
"""

import os
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
If the answer is not in the context, say so clearly.

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

# ── Lazy singletons ───────────────────────────────────────────────────────────
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

# ── Request models ────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question:    str     = Field(..., min_length=3)
    class_name:  str     = Field(..., example="class_10")
    subject:     str     = Field(..., example="mathematics")
    max_context: int     = Field(2000, ge=500, le=5000)
    answer_mode: Literal["detailed","brief","one_line","mcq","true_false"] = "detailed"

class WorksheetRequest(BaseModel):
    class_name:   str        = Field(..., example="class_10")
    subject:      str        = Field(..., example="mathematics")
    topics:       list[str]  = Field(..., min_length=1)
    difficulty:   Literal["easy","medium","hard","mixed"] = "medium"
    question_types: list[Literal["mcq","short","truefalse","fillblank","long"]] = ["mcq","short"]
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
        retriever = get_retriever()
        results   = retriever.retrieve_and_rerank(
            req.question,
            class_filter=req.class_name,
            subject_filter=req.subject
        )
    except Exception as e:
        yield sse({"type":"error","message":f"Retrieval error: {e}"}); return

    if not results:
        yield sse({"type":"error","message":"No relevant content found. Make sure PDFs are ingested."}); return

    # Extract docs
    top_docs = [r["doc"] for r in results]

    # Compute overall confidence (weighted average)
    console.print(f" results {results}")
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
    lang_ins = LANGUAGE_INSTRUCTIONS.get(req.subject, "Respond in English.")
    mode_ins = ANSWER_MODE_INSTRUCTIONS.get(
        req.answer_mode, ANSWER_MODE_INSTRUCTIONS["detailed"]
    )

    prompt = BASE_PROMPT.format(
        context=context,
        question=req.question,
        class_name=req.class_name.replace("_"," ").title(),
        subject=req.subject.replace("_"," ").title(),
        language_instruction=lang_ins,
        answer_mode_instruction=mode_ins,
    )

    yield sse({"type":"status","message":"Generating answer…"})
    await asyncio.sleep(0)

    # ── Stream tokens ─────────────────────────────────────────────────────────
    try:
        from langchain_core.messages import HumanMessage
        async for chunk in get_llm().astream([HumanMessage(content=prompt)]):
            token = chunk.content
            if token:
                visible = re.sub(r"<think>.*", "", token, flags=re.DOTALL)
                if visible:
                    yield sse({"type":"token","data":visible})
                    await asyncio.sleep(0)
    except Exception as e:
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
WORKSHEET_PROMPT = """You are an expert CBSE question paper setter for {class_name}, subject: {subject}.

{language_instruction}

Use ONLY the context below from NCERT textbooks to create worksheet questions.
If the context is insufficient for some question types, draw from closely related NCERT concepts.

TASK: Generate exactly {num_questions} worksheet questions on the topic(s): {topics}
Difficulty: {difficulty}
Question types to include (distribute evenly): {question_types}
{extra}

STRICT JSON OUTPUT — return ONLY this structure, no markdown fences, no preamble:
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
          "options": ["A. option1", "B. option2", "C. option3", "D. option4"]
        }},
        {{
          "type": "short",
          "text": "Short answer question here"
        }},
        {{
          "type": "truefalse",
          "text": "Statement for true/false"
        }},
        {{
          "type": "fillblank",
          "text": "Sentence with ___ for blank(s)"
        }},
        {{
          "type": "long",
          "text": "Essay or long answer question"
        }}
      ]
    }}
  ]
}}

Rules:
- Group questions by type into separate sections
- MCQ must have exactly 4 options labeled A–D, one must be correct
- Fill-in-the-blank must use ___ in the sentence for each blank
- Questions must be directly based on the CONTEXT below
- Age-appropriate for {class_name}

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
}

async def worksheet_stream(req: WorksheetRequest) -> AsyncGenerator[str, None]:
    if req.class_name not in SUPPORTED_CLASSES:
        yield sse({"type":"error","message":f"Invalid class: {req.class_name}"}); return
    if req.subject not in SUPPORTED_SUBJECTS:
        yield sse({"type":"error","message":f"Invalid subject: {req.subject}"}); return
    if not req.question_types:
        yield sse({"type":"error","message":"Select at least one question type."}); return

    yield sse({"type":"status","message":"Searching NCERT textbooks for relevant content…"})
    await asyncio.sleep(0)

    # Retrieve context for each topic and merge
    try:
        retriever = get_retriever()
        all_results = []
        seen_keys = set()
        for topic in req.topics:
            results = retriever.retrieve_and_rerank(
                topic,
                class_filter=req.class_name,
                subject_filter=req.subject
            )
            for r in results:
                doc = r["doc"]
                key = f"{doc.metadata.get('source','')}:{doc.metadata.get('page','')}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(r)
    except Exception as e:
        yield sse({"type":"error","message":f"Retrieval error: {e}"}); return

    if not all_results:
        yield sse({"type":"error","message":"No relevant content found. Make sure PDFs are ingested for this class/subject."}); return

    # Overall confidence
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

    # Send sources
    seen = set()
    sources = []
    for r in all_results:
        doc  = r["doc"]
        src  = doc.metadata.get("source","Unknown")
        page = doc.metadata.get("page","?")
        k    = f"{src}-{page}"
        if k not in seen:
            seen.add(k)
            sources.append({
                "file":       os.path.basename(src),
                "page":       page,
                "preview":    doc.page_content[:100].replace("\n"," "),
                "confidence": r["confidence"],
                "label":      r["label"],
                "color":      confidence_color(r["confidence"]),
            })
    yield sse({"type":"sources","data":sources})
    await asyncio.sleep(0)

    # Build context from top docs
    top_docs = [r["doc"] for r in all_results]
    context_parts, total_len = [], 0
    for doc in top_docs:
        if total_len + len(doc.page_content) > req.max_context: break
        context_parts.append(doc.page_content)
        total_len += len(doc.page_content)
    context = "\n\n---\n\n".join(context_parts)

    lang_ins = LANGUAGE_INSTRUCTIONS.get(req.subject, "Respond in English.")

    difficulty_map = {
        "easy":   "Easy — basic recall and definitions",
        "medium": "Medium — application and understanding",
        "hard":   "Hard — analysis, inference, and higher-order thinking",
        "mixed":  "Mixed — a blend of easy, medium, and hard questions",
    }
    qtype_labels = {
        "mcq":"Multiple Choice","short":"Short Answer",
        "truefalse":"True/False","fillblank":"Fill in the Blank","long":"Long Answer/Essay"
    }

    prompt = WORKSHEET_PROMPT.format(
        class_name=req.class_name.replace("_"," ").title(),
        subject=req.subject.replace("_"," ").title(),
        language_instruction=lang_ins,
        num_questions=req.num_questions,
        topics=", ".join(req.topics),
        difficulty=difficulty_map.get(req.difficulty, req.difficulty),
        question_types=", ".join(qtype_labels.get(t, t) for t in req.question_types),
        extra=f"Extra instructions: {req.extra_instructions}" if req.extra_instructions else "",
        context=context,
    )

    yield sse({"type":"status","message":"Generating worksheet questions from textbook content…"})
    await asyncio.sleep(0)

    # Stream LLM output and collect full response
    full_response = ""
    try:
        from langchain_core.messages import HumanMessage
        async for chunk in get_llm().astream([HumanMessage(content=prompt)]):
            token = chunk.content
            if token:
                full_response += token
                yield sse({"type":"progress","data":token})
                await asyncio.sleep(0)
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
        "llm_provider":   os.getenv("LLM_PROVIDER","groq"),
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
        return {"status":"ok","filename":file.filename,"chunks":chunks}

    except Exception as e:
        raise HTTPException(500, f"Ingestion error: {str(e)}")

@app.post("/api/extract-questions")
async def extract_questions(
    file:       UploadFile = File(...),
    class_name: str        = Form(...),
    subject:    str        = Form(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = ""
        try:
            import fitz
            fitz.TOOLS.mupdf_display_errors(False)
            fitz.TOOLS.mupdf_display_warnings(False)
            doc = fitz.open(tmp_path)
            for page in doc:
                try: text += page.get_text("text") + "\n"
                except: pass
            doc.close()
        except Exception:
            import pypdf
            reader = pypdf.PdfReader(tmp_path, strict=False)
            for page in reader.pages:
                try: text += (page.extract_text() or "") + "\n"
                except: pass

        if not text.strip():
            raise HTTPException(422, "Could not extract text from PDF")

        extraction_prompt = f"""Extract ALL questions from this question paper.
For each question return JSON with "text" (exact question) and "type" (mcq/true_false/short/long/fill).
Return ONLY a JSON array, no markdown, no explanation:
[{{"text":"...","type":"short"}}]

PAPER:
{text[:6000]}"""

        from langchain_core.messages import HumanMessage
        response = await get_llm().ainvoke([HumanMessage(content=extraction_prompt)])
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
        raise HTTPException(422, "Could not parse questions — try a cleaner PDF")
    except Exception as e:
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
