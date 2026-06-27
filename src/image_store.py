import os
import json
import fitz  # PyMuPDF
from rich.console import Console

console = Console()

IMAGE_STORE_DIR     = "image_store"
IMAGE_METADATA_FILE = "image_store/metadata.json"

def ensure_image_store():
    os.makedirs(IMAGE_STORE_DIR, exist_ok=True)

def load_image_metadata() -> dict:
    if os.path.exists(IMAGE_METADATA_FILE):
        with open(IMAGE_METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_image_metadata(metadata: dict):
    ensure_image_store()
    with open(IMAGE_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def render_page_as_image(
    doc,
    page_num: int,
    pdf_info: dict,
    dpi: int = 150
) -> dict | None:
    """
    Render a full PDF page as a PNG image.
    This captures vector diagrams, text-based figures, and everything
    visible on the page — much better than extracting embedded images.
    """
    try:
        page = doc[page_num]

        # Check if page has meaningful visual content
        # Pages with only a little text and some drawings = diagram page
        text        = page.get_text("text").strip()
        blocks      = page.get_text("blocks")
        drawings    = page.get_drawings()
        images      = page.get_images(full=True)

        has_drawings = len(drawings) > 5       # vector diagrams
        has_images   = len(images)   > 0       # embedded bitmaps
        text_length  = len(text)

        # Skip pages that are pure text with no visuals
        # A diagram page typically has drawings/images + some text (caption)
        if not has_drawings and not has_images:
            return None

        # Skip pages with too much text — likely text-only pages
        if text_length > 2000 and not has_images:
            return None

        # Render page to image at specified DPI
        mat    = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is base DPI
        pixmap = page.get_pixmap(matrix=mat, alpha=False)

        # Save as PNG
        img_filename = (
            f"{pdf_info['class']}_"
            f"{pdf_info['subject']}_"
            f"{pdf_info['filename'].replace('.pdf', '')}_"
            f"page{page_num:03d}.png"
        )
        img_path = os.path.join(IMAGE_STORE_DIR, img_filename)
        pixmap.save(img_path)

        # Get page text as caption context
        caption = text[:300].replace("\n", " ").strip()

        return {
            "filename":     img_filename,
            "path":         img_path,
            "class":        pdf_info["class"],
            "subject":      pdf_info["subject"],
            "source_pdf":   pdf_info["filename"],
            "page":         page_num,
            "width":        pixmap.width,
            "height":       pixmap.height,
            "has_drawings": has_drawings,
            "has_images":   has_images,
            "caption":      caption
        }

    except Exception as e:
        console.print(f"   [dim]Page {page_num} render failed: {e}[/dim]")
        return None

def index_pdf_images(pdf_info: dict) -> int:
    """
    Render diagram pages from a PDF and store them.
    Returns number of images extracted.
    """
    ensure_image_store()
    metadata = load_image_metadata()
    pdf_key  = pdf_info["key"]

    # Skip if already extracted
    if pdf_key in metadata.get("pdfs_processed", []):
        console.print(f"   [dim]Images already extracted — skipping[/dim]")
        return 0

    extracted = []

    try:
        doc = fitz.open(pdf_info["path"])
        console.print(
            f"   [dim]Scanning {len(doc)} pages for diagrams...[/dim]",
            end=""
        )

        for page_num in range(len(doc)):
            result = render_page_as_image(doc, page_num, pdf_info, dpi=150)
            if result:
                extracted.append(result)

        doc.close()
        console.print(
            f" [green]✓ {len(extracted)} diagram pages found[/green]"
        )

    except Exception as e:
        console.print(f"\n   [yellow]⚠ Image extraction failed: {e}[/yellow]")
        return 0

    # Save metadata
    if "images" not in metadata:
        metadata["images"] = []
    if "pdfs_processed" not in metadata:
        metadata["pdfs_processed"] = []

    metadata["images"].extend(extracted)
    metadata["pdfs_processed"].append(pdf_key)
    save_image_metadata(metadata)

    return len(extracted)

def find_relevant_images(
    class_name:  str,
    subject:     str,
    source_docs: list,
    max_images:  int = 2
) -> list:
    """
    Find rendered page images that match the source doc pages.
    Matches by class, subject, source PDF and page number proximity.
    """
    metadata = load_image_metadata()
    if not metadata.get("images"):
        return []

    all_images = metadata["images"]

    # Get pages referenced in source docs
    referenced = []
    for doc in source_docs:
        src  = doc.metadata.get("source", "")
        page = doc.metadata.get("page", -1)
        if src and page >= 0:
            referenced.append({"source": src, "page": page})

    if not referenced:
        return []

    # Score images by page proximity to source docs
    scored = []
    for img in all_images:

        # Must match class and subject
        if img["class"] != class_name or img["subject"] != subject:
            continue

        score = 0
        for ref in referenced:
            if img["source_pdf"] == ref["source"]:
                diff = abs(img["page"] - ref["page"])
                if diff == 0:
                    score += 10    # exact same page
                elif diff == 1:
                    score += 6     # adjacent page
                elif diff == 2:
                    score += 3     # 2 pages away
                elif diff <= 5:
                    score += 1     # nearby

        # Boost pages with actual drawings (vector diagrams)
        if img.get("has_drawings"):
            score += 2

        if score > 0:
            scored.append((score, img))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [img for _, img in scored[:max_images]]

def remove_pdf_images(pdf_key: str):
    """Remove all images for a specific PDF from image store."""
    metadata = load_image_metadata()

    if not metadata.get("images"):
        return

    # Get source_pdf name from key (class/subject/filename.pdf)
    source_pdf = pdf_key.split("/")[-1]

    # Remove image files
    removed = 0
    kept    = []
    for img in metadata["images"]:
        if img["source_pdf"] == source_pdf:
            try:
                if os.path.exists(img["path"]):
                    os.remove(img["path"])
                removed += 1
            except Exception:
                pass
        else:
            kept.append(img)

    # Update metadata
    metadata["images"] = kept
    if pdf_key in metadata.get("pdfs_processed", []):
        metadata["pdfs_processed"].remove(pdf_key)

    save_image_metadata(metadata)

    if removed:
        console.print(
            f"   [dim]Removed {removed} old images for {source_pdf}[/dim]"
        )