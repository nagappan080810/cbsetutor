"""
backfill_concept_graph.py
──────────────────────────
One-time (or re-run-anytime) backfill for PDFs that were ingested into
Chroma before concept_graph.py existed. Does NOT touch ChromaDB and does
NOT re-embed anything -- it only re-reads the original PDF files already
sitting under DATA_DIR and runs them through the same extraction pass
/api/upload now runs automatically for new uploads.

Safe to re-run: merge_and_save_chapter() merges into existing graph
nodes (extends key_facts/misconceptions, dedups) rather than overwriting.

Usage:
    python scripts/backfill_concept_graph.py
    python scripts/backfill_concept_graph.py --class class_10 --subject science
    python scripts/backfill_concept_graph.py --file data/class_10/science/ch3.pdf --class class_10 --subject science
"""
import os
import sys
import time
import argparse
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import concept_graph          # noqa: E402
from src.rag_chain import get_rag_chain  # noqa: E402

console = Console()
console_lock = threading.Lock()  # rich Console isn't guaranteed thread-safe for interleaved writes

# PyMuPDF (fitz) is NOT thread-safe: fitz.TOOLS is global mutable state and
# the underlying MuPDF font/glyph cache is shared across every fitz.open()
# call regardless of which thread makes it. Calling fitz concurrently from
# ThreadPoolExecutor workers (as BACKFILL_CONCURRENCY does) causes silent
# native segfaults with no Python traceback -- this bit us at
# CONCEPT_GRAPH_CONCURRENCY=6 (see backfillout6.log: crash happens right
# after the 6 filenames print, before any "extracting concepts" line --
# i.e. inside _extract_pdf_text, not the LLM phase). PDF text extraction is
# fast and local, so serializing just this part costs almost nothing; the
# actual slow/parallelizable work (LLM extraction calls) stays fully
# concurrent, unaffected by this lock.
_fitz_lock = threading.Lock()

DATA_DIR = os.getenv("DATA_DIR", "./data")
# How many (class, subject) graphs to process at once. Chapters WITHIN the
# same subject stay serialized (see _subject_locks below) because
# build_from_chapter_text does a read-modify-write on that subject's single
# graph.json -- concurrent writers there would race and silently drop a
# chapter's concepts. Across-subject parallelism is safe since each subject
# has its own graph.json. Default of 8 is conservative for NVIDIA NIM's free
# tier (32-concurrent cap) -- raise it if you're on Groq/DeepSeek or Ollama,
# where limits are higher or (for local Ollama) governed by your own GPU
# instead of a provider quota. Tune via env, no code change needed.
BACKFILL_CONCURRENCY = int(os.getenv("CONCEPT_GRAPH_CONCURRENCY", 8))
# keep in sync with api.py's SUPPORTED_CLASSES / SUPPORTED_SUBJECTS
SUPPORTED_CLASSES = ["class_8", "class_9", "class_10"]
SUPPORTED_SUBJECTS = [
    "mathematics", "science", "socialscience",
    "english", "hindi", "kannada", "tamil", "sanskrit",
]


def _extract_pdf_text(pdf_path: str) -> str:
    """Same fitz-first/pypdf-fallback logic as api.py's _extract_pdf_text --
    duplicated here (not imported) so this script has zero dependency on
    api.py / FastAPI app startup and can run standalone from a cron job."""
    text = ""
    try:
        import fitz
        # fitz.open()/TOOLS calls must be serialized across threads -- see
        # the _fitz_lock comment above. pypdf (the fallback) is pure Python
        # and doesn't need this.
        with _fitz_lock:
            fitz.TOOLS.mupdf_display_errors(False)
            fitz.TOOLS.mupdf_display_warnings(False)
            doc = fitz.open(pdf_path)
            try:
                for page in doc:
                    try: text += page.get_text("text") + "\n"
                    except Exception: pass
            finally:
                doc.close()
    except Exception:
        import pypdf
        reader = pypdf.PdfReader(pdf_path, strict=False)
        for page in reader.pages:
            try: text += (page.extract_text() or "") + "\n"
            except Exception: pass
    return text


def _process_one(pdf_path: Path, cls: str, subj: str, llm_raw, subject_lock) -> dict:
    """Run one PDF through extraction, then merge into that subject's graph.
    Extraction (the slow, LLM-bound part) runs with NO lock held, so it can
    overlap freely with other chapters' extraction -- in the same subject
    or a different one. The subject_lock is only acquired for the merge+save
    step, which is fast, local, and the one part that races if two chapters
    from the same subject try to write graph.json at once."""
    with console_lock:
        console.print(f"[cyan]→ {cls}/{subj}/{pdf_path.name}[/cyan]")
    try:
        text = _extract_pdf_text(str(pdf_path))
        extracted = concept_graph.extract_concepts_from_chapter(
            chapter_text=text,
            chapter=pdf_path.name,
            llm_raw=llm_raw,
        )
        with subject_lock:
            result = concept_graph.merge_and_save_chapter(
                extracted,
                class_name=cls,
                subject=subj,
                chapter=pdf_path.name,
                source=pdf_path.name,
            )
        with console_lock:
            console.print(f"   {cls}/{subj}/{pdf_path.name} → {result}")
    except Exception as e:
        result = {"status": "failed", "error": str(e)}
        with console_lock:
            console.print(f"[red]   {cls}/{subj}/{pdf_path.name} failed: {e}[/red]")

    result["pdf_path"] = pdf_path
    result["cls"] = cls
    result["subj"] = subj
    return result


def _retry_one_chapter(pdf_path: Path, cls: str, subj: str, llm_raw, subject_lock,
                        failed_window_indices: list) -> dict:
    """Re-extracts ONLY the window indices that failed last pass, instead of
    the whole chapter. The successful windows' concepts from the first pass
    are already merged into graph.json (merge_and_save_chapter merges even
    partial results) -- this just fills in the gap. Re-reading the PDF text
    is cheap/local; only the LLM calls for the specific failed windows are
    re-run, cutting redundant work by up to windows_total-1x per retry."""
    with console_lock:
        console.print(
            f"[cyan]→ {cls}/{subj}/{pdf_path.name} "
            f"(retrying window(s) {failed_window_indices})[/cyan]"
        )
    try:
        text = _extract_pdf_text(str(pdf_path))
        extracted = concept_graph.retry_failed_windows(
            chapter_text=text,
            chapter=pdf_path.name,
            llm_raw=llm_raw,
            failed_window_indices=failed_window_indices,
        )
        with subject_lock:
            result = concept_graph.merge_and_save_chapter(
                extracted,
                class_name=cls,
                subject=subj,
                chapter=pdf_path.name,
                source=pdf_path.name,
            )
        with console_lock:
            console.print(f"   {cls}/{subj}/{pdf_path.name} → {result}")
    except Exception as e:
        result = {"status": "failed", "error": str(e)}
        with console_lock:
            console.print(f"[red]   {cls}/{subj}/{pdf_path.name} retry failed: {e}[/red]")

    result["pdf_path"] = pdf_path
    result["cls"] = cls
    result["subj"] = subj
    return result


def backfill(class_filter=None, subject_filter=None, file_filter=None,
             retry_partial=True, retry_cooldown_seconds=60, max_retry_passes=2):
    if not concept_graph.CONCEPT_GRAPH_ENABLED:
        console.print(
            "[red]concept_graph is disabled (CONCEPT_GRAPH_ENABLED=false, or "
            "networkx isn't installed) -- nothing to do.[/red]"
        )
        return

    llm_raw = get_rag_chain().llm_raw
    classes = [class_filter] if class_filter else SUPPORTED_CLASSES
    subjects = [subject_filter] if subject_filter else SUPPORTED_SUBJECTS

    # One lock per (class, subject) -- guards only merge_and_save_chapter's
    # graph.json read-modify-write, not extraction. Lazily created so we
    # don't need to know the (class, subject) universe up front.
    subject_locks = defaultdict(threading.Lock)

    todo = []  # list of (pdf_path, cls, subj)
    for cls in classes:
        for subj in subjects:
            subject_dir = Path(DATA_DIR) / cls / subj
            if not subject_dir.exists():
                continue
            pdfs = [Path(file_filter)] if file_filter else sorted(subject_dir.glob("*.pdf"))
            for pdf_path in pdfs:
                if not pdf_path.exists():
                    console.print(f"[yellow]skip (not found): {pdf_path}[/yellow]")
                    continue
                todo.append((pdf_path, cls, subj))
                if file_filter:
                    break
            if file_filter:
                break
        if file_filter:
            break

    total_files = len(todo)
    ok, partial, failed = {}, {}, {}  # keyed by pdf_path -> result dict

    for pass_num in range(1 + (max_retry_passes if retry_partial else 0)):
        if pass_num == 0:
            batch = [(pdf_path, cls, subj, None) for pdf_path, cls, subj in todo]
        else:
            # only re-attempt the SPECIFIC windows still stuck in "partial"
            # from last pass -- not the whole chapter (see _retry_one_chapter)
            batch = [
                (r["pdf_path"], r["cls"], r["subj"], r.get("failed_window_indices") or [])
                for r in partial.values()
            ]
            if not batch:
                break
            console.print(
                f"\n[bold yellow]Retry pass {pass_num}: {len(batch)} partial "
                f"chapter(s), waiting {retry_cooldown_seconds}s for rate-limit "
                f"window to reset...[/bold yellow]"
            )
            time.sleep(retry_cooldown_seconds)
            partial = {}  # will be repopulated below with whatever's still partial

        with ThreadPoolExecutor(max_workers=BACKFILL_CONCURRENCY) as pool:
            if pass_num == 0:
                futures = {
                    pool.submit(_process_one, pdf_path, cls, subj, llm_raw, subject_locks[(cls, subj)]): pdf_path
                    for pdf_path, cls, subj, _ in batch
                }
            else:
                futures = {
                    pool.submit(_retry_one_chapter, pdf_path, cls, subj, llm_raw,
                                 subject_locks[(cls, subj)], failed_indices): pdf_path
                    for pdf_path, cls, subj, failed_indices in batch
                }
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"status": "failed", "error": str(e), "pdf_path": pdf_path,
                               "cls": None, "subj": None}
                status = result.get("status")
                if status == "ok":
                    ok[pdf_path] = result
                    failed.pop(pdf_path, None)
                elif status == "partial":
                    partial[pdf_path] = result
                else:
                    failed[pdf_path] = result

    console.print(
        f"\n[bold]Backfill done:[/bold] {len(ok)} fully ok, {len(partial)} "
        f"still partial, {len(failed)} failed, out of {total_files} total"
    )
    if partial:
        console.print("[yellow]Still partial (re-run later, e.g. with a smaller "
                       "CONCEPT_GRAPH_CONCURRENCY or once rate limits reset):[/yellow]")
        for pdf_path, r in partial.items():
            console.print(f"  - {r['cls']}/{r['subj']}/{pdf_path.name}  "
                           f"({r.get('windows_dropped')}/{r.get('windows_total')} windows dropped)")
    if failed:
        console.print("[red]Failed (raised an exception, not rate-limit related):[/red]")
        for pdf_path, r in failed.items():
            console.print(f"  - {r['cls']}/{r['subj']}/{pdf_path.name}: {r.get('error')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="class_name", default=None, help="limit to one class, e.g. class_10")
    ap.add_argument("--subject", default=None, help="limit to one subject, e.g. science")
    ap.add_argument("--file", default=None, help="process a single PDF path instead of scanning a directory")
    ap.add_argument("--no-retry", dest="retry_partial", action="store_false",
                     help="don't auto-retry partial (rate-limited) chapters")
    ap.add_argument("--retry-cooldown", type=int, default=60,
                     help="seconds to wait before retrying partial chapters (default: 60)")
    ap.add_argument("--max-retry-passes", type=int, default=2,
                     help="how many retry passes to attempt on still-partial chapters (default: 2)")
    args = ap.parse_args()
    backfill(args.class_name, args.subject, args.file,
             retry_partial=args.retry_partial,
             retry_cooldown_seconds=args.retry_cooldown,
             max_retry_passes=args.max_retry_passes)
