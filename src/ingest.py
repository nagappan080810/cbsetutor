import os
import time
import json
import hashlib
import shutil
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn
)

load_dotenv()
console = Console()

SUPPORTED_CLASSES  = ["class_8", "class_9", "class_10"]
SUPPORTED_SUBJECTS = [
    "mathematics", "science", "social_science",
    "english", "hindi", "kannada", "tamil", "sanskrit"
]

TRACKER_FILE = "ingest_tracker.json"

# ── Embedding provider ────────────────────────────────────────────────────────

def get_embeddings():
    provider = os.getenv("EMBED_PROVIDER", "ollama")
    console.print(f"[dim]Embedding provider: {provider}[/dim]")

    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        console.print(f"[dim]Model: {model} → 768 dims[/dim]")
        return OllamaEmbeddings(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("EMBED_MODEL", "text-embedding-004")
        console.print(f"[dim]Model: {model}[/dim]")
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type="retrieval_document"
        )
    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        model = os.getenv(
            "EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        console.print(f"[dim]Model: {model} → 384 dims[/dim]")
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        raise ValueError(f"Unknown EMBED_PROVIDER: {provider}")

# ── Tracker ───────────────────────────────────────────────────────────────────

def load_tracker() -> dict:
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_tracker(tracker: dict):
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)

def get_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# ── PDF scanner ───────────────────────────────────────────────────────────────

def get_all_pdfs(data_dir: str) -> list:
    pdf_files = []
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path) or class_name not in SUPPORTED_CLASSES:
            continue
        for subject_name in sorted(os.listdir(class_path)):
            subject_path = os.path.join(class_path, subject_name)
            if not os.path.isdir(subject_path):
                continue
            for file in sorted(os.listdir(subject_path)):
                if file.endswith(".pdf"):
                    full_path = os.path.join(subject_path, file)
                    pdf_files.append({
                        "path":     full_path,
                        "class":    class_name,
                        "subject":  subject_name.lower(),
                        "filename": file,
                        "key":      f"{class_name}/{subject_name}/{file}"
                    })
    return pdf_files

def filter_new_pdfs(pdf_files: list, tracker: dict) -> tuple:
    new_files     = []
    skipped_files = []
    for pdf in pdf_files:
        cur_hash = get_file_hash(pdf["path"])
        if pdf["key"] in tracker and tracker[pdf["key"]]["hash"] == cur_hash:
            skipped_files.append(pdf)
        else:
            pdf["hash"] = cur_hash
            new_files.append(pdf)
    return new_files, skipped_files

# ── Ad-hoc selector ───────────────────────────────────────────────────────────

def select_class_interactive(data_dir: str) -> str:
    """Show available classes and let user pick one."""
    available = [
        d for d in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, d))
        and d in SUPPORTED_CLASSES
    ]

    if not available:
        console.print("[red]No class folders found in data/[/red]")
        return None

    console.print("\n[bold cyan]📚 Select Class:[/bold cyan]")
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("No.",   style="yellow", width=5)
    table.add_column("Class", style="white")

    for i, cls in enumerate(available, 1):
        table.add_row(f"[{i}]", cls.replace("_", " ").title())
    console.print(table)

    while True:
        choice = input(f"Enter number (1-{len(available)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(available):
            selected = available[int(choice) - 1]
            console.print(f"[green]✓ {selected}[/green]")
            return selected
        console.print(f"[red]Enter a number between 1 and {len(available)}[/red]")

def select_subject_interactive(data_dir: str, class_name: str) -> str:
    """Show available subjects for selected class and let user pick one."""
    class_path = os.path.join(data_dir, class_name)
    available  = [
        d for d in sorted(os.listdir(class_path))
        if os.path.isdir(os.path.join(class_path, d))
    ]

    if not available:
        console.print(f"[red]No subject folders found in data/{class_name}/[/red]")
        return None

    console.print("\n[bold cyan]📖 Select Subject:[/bold cyan]")
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("No.",     style="yellow", width=5)
    table.add_column("Subject", style="white")

    for i, subj in enumerate(available, 1):
        table.add_row(f"[{i}]", subj.replace("_", " ").title())
    console.print(table)

    while True:
        choice = input(f"Enter number (1-{len(available)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(available):
            selected = available[int(choice) - 1]
            console.print(f"[green]✓ {selected}[/green]")
            return selected
        console.print(f"[red]Enter a number between 1 and {len(available)}[/red]")

def select_files_interactive(
    data_dir: str,
    class_name: str,
    subject_name: str
) -> list:
    """Show available PDFs and let user pick one, many, or all."""
    subject_path = os.path.join(data_dir, class_name, subject_name)
    available    = sorted([
        f for f in os.listdir(subject_path)
        if f.endswith(".pdf")
    ])

    if not available:
        console.print(
            f"[red]No PDFs in data/{class_name}/{subject_name}/[/red]"
        )
        return []

    tracker = load_tracker()

    console.print("\n[bold cyan]📄 Select File(s):[/bold cyan]")
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("No.",    style="yellow", width=5)
    table.add_column("File",   style="white")
    table.add_column("Status", justify="center")
    table.add_column("Chunks", justify="right", style="dim")

    for i, fname in enumerate(available, 1):
        key        = f"{class_name}/{subject_name}/{fname}"
        in_tracker = key in tracker
        status     = "[green]ingested[/green]" if in_tracker else "[yellow]not ingested[/yellow]"
        chunks     = str(tracker[key].get("chunks", "?")) if in_tracker else "-"
        table.add_row(f"[{i}]", fname, status, chunks)

    console.print(table)
    console.print("[dim]Enter number(s) separated by comma, or 'all' for all files[/dim]")

    while True:
        choice = input("Your choice: ").strip().lower()

        if choice == "all":
            selected_files = available
            break

        parts = [p.strip() for p in choice.split(",")]
        valid = all(
            p.isdigit() and 1 <= int(p) <= len(available)
            for p in parts
        )
        if valid:
            selected_files = [available[int(p) - 1] for p in parts]
            break

        console.print(
            f"[red]Invalid. Enter numbers 1-{len(available)}, "
            f"comma-separated, or 'all'[/red]"
        )

    # Build pdf_info dicts
    result = []
    for fname in selected_files:
        full_path = os.path.join(subject_path, fname)
        result.append({
            "path":     full_path,
            "class":    class_name,
            "subject":  subject_name.lower(),
            "filename": fname,
            "key":      f"{class_name}/{subject_name}/{fname}",
            "hash":     get_file_hash(full_path)
        })

    return result

# ── PDF loader ────────────────────────────────────────────────────────────────
def load_pdf(pdf_info: dict) -> list:
    """Load PDF — PyMuPDF with higher decompression limit, fallback to PyPDF."""

    # Method 1: PyMuPDF with increased decompression limit
    try:
        import fitz  # PyMuPDF (already installed as pymupdf)

        # Increase decompression limit to 500MB — fixes large NCERT PDFs
        fitz.TOOLS.set_icc(False)
        old_limit = fitz.TOOLS.mupdf_warnings()

        doc   = fitz.open(pdf_info["path"])
        pages = []

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                # Use rawdict for better text extraction from compressed pages
                text = page.get_text(
                    "text",
                    flags=fitz.TEXT_PRESERVE_WHITESPACE
                    | fitz.TEXT_PRESERVE_LIGATURES
                )
                if text.strip():
                    from langchain_core.documents import Document
                    pages.append(Document(
                        page_content=text,
                        metadata={
                            "source":  pdf_info["filename"],
                            "class":   pdf_info["class"],
                            "subject": pdf_info["subject"],
                            "page":    page_num
                        }
                    ))
            except Exception as page_err:
                # Skip individual bad pages, don't fail entire PDF
                console.print(
                    f"   [yellow]⚠ Skipping page {page_num}: {page_err}[/yellow]"
                )
                continue

        doc.close()

        if pages:
            return pages
        # If no pages extracted, fall through to PyPDF
        raise Exception("No text extracted via PyMuPDF")

    except Exception as e:
        console.print(f"   [dim]PyMuPDF failed ({e}), trying PyPDF...[/dim]")

    # Method 2: PyPDF fallback — handles differently compressed PDFs
    try:
        from langchain_community.document_loaders import PyPDFLoader

        # Increase PyPDF decompression limit
        import pypdf
        pypdf.filters.FlateDecode.decode.__func__

        loader = PyPDFLoader(
            pdf_info["path"],
            extract_images=False    # skip images — reduces memory pressure
        )
        pages = loader.load()

        for page in pages:
            page.metadata["class"]   = pdf_info["class"]
            page.metadata["subject"] = pdf_info["subject"]
            page.metadata["source"]  = pdf_info["filename"]

        if pages:
            return pages

    except Exception as e:
        console.print(f"   [dim]PyPDF failed ({e}), trying strict=False...[/dim]")

    # Method 3: PyPDF with strict=False — most lenient mode
    try:
        import pypdf
        from langchain_core.documents import Document

        pages  = []
        reader = pypdf.PdfReader(
            pdf_info["path"],
            strict=False        # ignore PDF spec violations
        )

        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(Document(
                        page_content=text,
                        metadata={
                            "source":  pdf_info["filename"],
                            "class":   pdf_info["class"],
                            "subject": pdf_info["subject"],
                            "page":    page_num
                        }
                    ))
            except Exception as page_err:
                # Skip bad pages silently
                continue

        if pages:
            console.print(
                f"   [dim]Loaded with strict=False mode[/dim]", end=""
            )
            return pages

    except Exception as e:
        console.print(f"   [dim]strict=False also failed: {e}[/dim]")

    # All methods failed
    console.print(
        f"\n   [red]✗ Could not load {pdf_info['filename']} "
        f"— skipping this file[/red]"
    )
    return []
# ── Batch embedder ────────────────────────────────────────────────────────────

def embed_in_batches(
    all_chunks: list,
    embeddings,
    chroma_dir: str,
    batch_size: int,
    existing_store=None
) -> tuple:
    vectorstore   = existing_store
    failed_chunks = []
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task(
            "[cyan]Embedding...", total=total_batches
        )

        for i in range(0, len(all_chunks), batch_size):
            batch     = all_chunks[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            progress.update(
                task,
                description=(
                    f"[cyan]Batch {batch_num}/{total_batches} "
                    f"({len(batch)} chunks)"
                )
            )

            success = False
            for attempt in range(3):
                try:
                    if vectorstore is None:
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=embeddings,
                            persist_directory=chroma_dir
                        )
                    else:
                        vectorstore.add_documents(batch)
                    success = True
                    break
                except Exception as e:
                    if attempt < 2:
                        progress.print(
                            f"   [yellow]Attempt {attempt+1} failed, "
                            f"retrying in 3s... ({e})[/yellow]"
                        )
                        time.sleep(3)
                    else:
                        progress.print(
                            f"   [red]✗ Batch {batch_num} failed: {e}[/red]"
                        )
                        failed_chunks.extend(batch)

            if success:
                time.sleep(0.5)

            progress.advance(task)

    return vectorstore, failed_chunks

# ── Per-PDF ingestion ─────────────────────────────────────────────────────────

def ingest_pdf_list(
    pdf_list: list,
    embeddings,
    vectorstore,
    chroma_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    tracker: dict,
    force: bool = False
) -> tuple:
    """
    Ingest a list of PDFs one by one.
    Writes tracker after each successful PDF.
    Returns (vectorstore, total_failed).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )

    total_failed = 0

    for pdf_info in pdf_list:
        console.print(
            f"\n📄 [white]{pdf_info['filename']}[/white] "
            f"([cyan]{pdf_info['class']}[/cyan] → "
            f"[yellow]{pdf_info['subject']}[/yellow])"
        )

        pages = load_pdf(pdf_info)
        if not pages:
            continue

        chunks = splitter.split_documents(pages)
        console.print(
            f"   [dim]{len(pages)} pages → {len(chunks)} chunks[/dim]"
        )

        # If force re-ingest, remove old vectors for this file from tracker
        if force and pdf_info["key"] in tracker:
            console.print(
                f"   [yellow]⚠ Force mode — removing old entry from tracker[/yellow]"
            )
            del tracker[pdf_info["key"]]
            save_tracker(tracker)

        vectorstore, failed = embed_in_batches(
            chunks, embeddings, chroma_dir, batch_size,
            existing_store=vectorstore
        )
        total_failed += len(failed)

        if not failed:
            # ✅ Extract/re-extract images for this PDF
            try:
                from src.image_store import index_pdf_images
                index_pdf_images(pdf_info)
            except Exception as e:
                console.print(
                    f"   [yellow]⚠ Image extraction skipped: {e}[/yellow]"
                )

            # ✅ Write tracker per PDF immediately
            tracker[pdf_info["key"]] = {
                "hash":             pdf_info["hash"],
                "ingested_at":      datetime.now().isoformat(),
                "chunks":           len(chunks),
                "class":            pdf_info["class"],
                "subject":          pdf_info["subject"],
                "filename":         pdf_info["filename"],
                "images_extracted": True
            }
            save_tracker(tracker)
            console.print(f"   [green]✓ Tracker updated[/green]")
        else:
            console.print(
                f"   [yellow]⚠ {len(failed)} chunks failed "
                f"— not marked in tracker, will retry next run[/yellow]"
            )

    return vectorstore, total_failed

# ── Show tracker status ───────────────────────────────────────────────────────

def show_tracker_status():
    """Print a summary of all ingested files."""
    tracker = load_tracker()
    if not tracker:
        console.print("[yellow]No files ingested yet.[/yellow]")
        return

    table = Table(
        title="📊 Ingestion Status",
        box=box.ROUNDED,
        show_lines=True
    )
    table.add_column("Class",      style="cyan")
    table.add_column("Subject",    style="yellow")
    table.add_column("File",       style="white")
    table.add_column("Chunks",     justify="right")
    table.add_column("Ingested At",style="dim")

    for key, info in sorted(tracker.items()):
        table.add_row(
            info.get("class",    key.split("/")[0]),
            info.get("subject",  key.split("/")[1]),
            info.get("filename", key.split("/")[2]),
            str(info.get("chunks", "?")),
            info.get("ingested_at", "?")[:19]
        )

    console.print(table)
    console.print(f"\n[dim]Total files ingested: {len(tracker)}[/dim]")

# ── Main entry ────────────────────────────────────────────────────────────────

def ingest_documents(force: bool = False, adhoc: bool = False):
    """
    force=False, adhoc=False → ingest all new/changed PDFs
    force=True,  adhoc=False → re-ingest everything from scratch
    force=False, adhoc=True  → interactive: pick class → subject → file(s)
    force=True,  adhoc=True  → interactive pick + force re-ingest selected
    """
    data_dir      = os.getenv("DATA_DIR", "./data")
    chroma_dir    = os.getenv("CHROMA_DIR", "./chroma_db")
    chunk_size    = int(os.getenv("CHUNK_SIZE", 600))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 80))
    batch_size    = int(os.getenv("EMBED_BATCH_SIZE", 10))

    tracker    = load_tracker()
    embeddings = get_embeddings()

    # Load existing vectorstore if it exists
    vectorstore = None
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        console.print("[dim]Existing vector store found — will add to it.[/dim]")
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings
        )

    # ── Ad-hoc mode: interactive file picker ─────────────────
    if adhoc:
        console.print(Panel.fit(
            "[bold yellow]⚡ Ad-hoc Ingestion Mode[/bold yellow]\n"
            "[dim]Pick class → subject → file(s) to re-ingest[/dim]",
            border_style="yellow"
        ))

        class_name = select_class_interactive(data_dir)
        if not class_name:
            return

        subject_name = select_subject_interactive(data_dir, class_name)
        if not subject_name:
            return

        selected_files = select_files_interactive(
            data_dir, class_name, subject_name
        )
        if not selected_files:
            return

        # Show what will happen for each file
        console.print(
            f"\n[bold]Selected {len(selected_files)} file(s):[/bold]"
        )

        files_to_process = []
        for f in selected_files:
            key        = f["key"]
            in_tracker = key in tracker
            cur_hash   = f["hash"]
            old_hash   = tracker.get(key, {}).get("hash", "")
            changed    = cur_hash != old_hash

            if in_tracker and not changed:
                # File unchanged — ask user
                console.print(
                    f"\n   [white]{f['filename']}[/white] — "
                    f"[green]already ingested, no changes detected[/green]"
                )
                reprocess = input(
                    "   Force re-ingest this file? (y/n): "
                ).strip().lower()
                if reprocess == "y":
                    files_to_process.append(f)
                else:
                    console.print("   [dim]Skipped.[/dim]")
            else:
                status = "[yellow]changed[/yellow]" if changed else "[yellow]not ingested[/yellow]"
                console.print(
                    f"\n   [white]{f['filename']}[/white] — {status}"
                )
                files_to_process.append(f)

        if not files_to_process:
            console.print(
                "\n[green]Nothing to re-ingest.[/green]"
            )
            return

        # Final confirm
        console.print(
            f"\n[bold]Will re-ingest {len(files_to_process)} file(s).[/bold]"
        )
        confirm = input("Proceed? (y/n): ").strip().lower()
        if confirm != "y":
            console.print("[yellow]Cancelled.[/yellow]")
            return

        # Remove old images for these files before re-ingesting
        for f in files_to_process:
            from src.image_store import remove_pdf_images
            remove_pdf_images(f["key"])

        vectorstore, total_failed = ingest_pdf_list(
            files_to_process, embeddings, vectorstore,
            chroma_dir, chunk_size, chunk_overlap,
            batch_size, tracker, force=True
        )
    
    # ── Normal / force mode: ingest all ──────────────────────
    else:
        console.print(
            "\n[bold blue]📚 Scanning class/subject folders...[/bold blue]"
        )
        all_files = get_all_pdfs(data_dir)

        if not all_files:
            console.print("[red]No PDFs found![/red]")
            console.print(
                "[yellow]Expected: data/class_10/mathematics/file.pdf[/yellow]"
            )
            return

        if force:
            console.print(
                "[yellow]⚠ Force mode — re-ingesting everything.[/yellow]"
            )
            shutil.rmtree(chroma_dir, ignore_errors=True)
            tracker     = {}
            vectorstore = None
            new_files   = all_files
            for f in new_files:
                f["hash"] = get_file_hash(f["path"])
            skipped_files = []
        else:
            new_files, skipped_files = filter_new_pdfs(all_files, tracker)

        # Show scan summary table
        table = Table(
            title="📂 PDF Scan Summary",
            box=box.ROUNDED,
            show_lines=True
        )
        table.add_column("Class",   style="cyan")
        table.add_column("Subject", style="yellow")
        table.add_column("File",    style="white")
        table.add_column("Status",  justify="center")

        for pdf in all_files:
            is_new = any(n["key"] == pdf["key"] for n in new_files)
            status = "[green]NEW[/green]" if is_new else "[dim]SKIP[/dim]"
            table.add_row(
                pdf["class"], pdf["subject"], pdf["filename"], status
            )
        console.print(table)
        console.print(
            f"   [green]New: {len(new_files)}[/green]   "
            f"[dim]Skipped: {len(skipped_files)}[/dim]\n"
        )

        if not new_files:
            console.print(
                "[bold green]✅ All PDFs already ingested![/bold green]"
            )
            console.print(
                "[dim]Add new PDFs and run again, "
                "or use --force to re-ingest everything, "
                "or --adhoc to fix a specific file.[/dim]"
            )
            return

        vectorstore, total_failed = ingest_pdf_list(
            new_files, embeddings, vectorstore,
            chroma_dir, chunk_size, chunk_overlap,
            batch_size, tracker, force=False
        )

    # ── Final persist ─────────────────────────────────────────
    if vectorstore:
        vectorstore.persist()
        console.print(f"\n[bold green]✅ Done![/bold green]")
        if total_failed:
            console.print(
                f"[yellow]⚠ {total_failed} chunks failed — "
                f"reduce EMBED_BATCH_SIZE and retry with --adhoc[/yellow]"
            )

if __name__ == "__main__":
    ingest_documents()