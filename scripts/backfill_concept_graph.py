"""
backfill_concept_graph.py
──────────────────────────
One-time (or re-run-anytime) backfill for PDFs that were ingested into
Chroma before concept_graph.py existed. Does NOT touch ChromaDB and does
NOT re-embed anything -- it only re-reads the original PDF files already
sitting under DATA_DIR and runs them through the same extraction pass
/api/upload now runs automatically for new uploads.

Safe to re-run: build_from_chapter_text() merges into existing graph
nodes (extends key_facts/misconceptions, dedups) rather than overwriting.

Usage:
    python scripts/backfill_concept_graph.py
    python scripts/backfill_concept_graph.py --class class_10 --subject science
    python scripts/backfill_concept_graph.py --file data/class_10/science/ch3.pdf --class class_10 --subject science
"""
import os
import sys
import argparse
from pathlib import Path
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import concept_graph          # noqa: E402
from src.rag_chain import get_rag_chain  # noqa: E402

console = Console()

DATA_DIR = os.getenv("DATA_DIR", "./data")
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
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.mupdf_display_warnings(False)
        doc = fitz.open(pdf_path)
        for page in doc:
            try: text += page.get_text("text") + "\n"
            except Exception: pass
        doc.close()
    except Exception:
        import pypdf
        reader = pypdf.PdfReader(pdf_path, strict=False)
        for page in reader.pages:
            try: text += (page.extract_text() or "") + "\n"
            except Exception: pass
    return text


def backfill(class_filter=None, subject_filter=None, file_filter=None):
    if not concept_graph.CONCEPT_GRAPH_ENABLED:
        console.print(
            "[red]concept_graph is disabled (CONCEPT_GRAPH_ENABLED=false, or "
            "networkx isn't installed) -- nothing to do.[/red]"
        )
        return

    llm_raw = get_rag_chain().llm_raw
    classes = [class_filter] if class_filter else SUPPORTED_CLASSES
    subjects = [subject_filter] if subject_filter else SUPPORTED_SUBJECTS

    total_files = total_ok = total_failed = 0

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
                total_files += 1
                console.print(f"[cyan]→ {cls}/{subj}/{pdf_path.name}[/cyan]")
                try:
                    text = _extract_pdf_text(str(pdf_path))
                    result = concept_graph.build_from_chapter_text(
                        chapter_text=text,
                        class_name=cls,
                        subject=subj,
                        chapter=pdf_path.name,
                        llm_raw=llm_raw,
                        source=pdf_path.name,
                    )
                    if result.get("status") == "ok":
                        total_ok += 1
                    console.print(f"   {result}")
                except Exception as e:
                    total_failed += 1
                    console.print(f"[red]   failed: {e}[/red]")
                if file_filter:
                    break
            if file_filter:
                break
        if file_filter:
            break

    console.print(
        f"\n[bold]Backfill done:[/bold] {total_ok}/{total_files} chapters "
        f"processed, {total_failed} failed"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="class_name", default=None, help="limit to one class, e.g. class_10")
    ap.add_argument("--subject", default=None, help="limit to one subject, e.g. science")
    ap.add_argument("--file", default=None, help="process a single PDF path instead of scanning a directory")
    args = ap.parse_args()
    backfill(args.class_name, args.subject, args.file)
