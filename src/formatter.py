import os
import re
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

def confidence_bar(pct: float, width: int = 20) -> Text:
    """Render a coloured ASCII progress bar for confidence."""
    filled = int((pct / 100) * width)
    empty  = width - filled

    if pct >= 85:
        color = "green"
    elif pct >= 60:
        color = "yellow"
    elif pct >= 40:
        color = "dark_orange"
    else:
        color = "red"

    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * empty,  style="dim")
    bar.append(f"  {pct:.1f}%", style=color + " bold")
    return bar

def display_images_terminal(images: list):
    if not images:
        return

    console.print("\n[bold magenta]🖼  Relevant Diagrams/Figures:[/bold magenta]")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("",      style="magenta", width=3)
    table.add_column("File",  style="white")
    table.add_column("Page",  style="dim", justify="center")
    table.add_column("Size",  style="dim", justify="right")

    for i, img in enumerate(images, 1):
        size_kb = img.get("size_bytes", 0) // 1024
        table.add_row(
            f"[{i}]",
            img["filename"],
            f"p.{img['page']}",
            f"{size_kb} KB"
        )

    console.print(table)
    console.print(
        "[dim]Images saved in: image_store/ — open them to view diagrams[/dim]"
    )

def format_answer(
    question:            str,
    answer:              str,
    source_docs:         list,
    images:              list = None,
    results:             list = None,   # list of {doc, score, confidence, label}
    overall_confidence:  dict = None    # {score, label, color}
):
    """Render answer with confidence scores in the terminal."""

    # Strip DeepSeek <think> tags
    clean_answer = re.sub(
        r"<think>.*?</think>", "", answer, flags=re.DOTALL
    ).strip()

    console.print(f"\n[bold cyan]❓ Question:[/bold cyan] {question}\n")

    # Overall confidence banner
    if overall_confidence:
        pct   = overall_confidence["score"]
        label = overall_confidence["label"]
        color = {"High": "green", "Medium": "yellow",
                 "Low": "dark_orange", "Very Low": "red"}.get(label, "white")
        console.print(
            f"[bold]Answer Confidence:[/bold]  ",
            end=""
        )
        console.print(confidence_bar(pct))
        console.print(
            f"[dim]Confidence indicates how well the retrieved sources match "
            f"your question.[/dim]\n"
        )

    # Answer panel
    console.print(Panel(
        Markdown(clean_answer),
        title="[bold green]📖 Answer[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))

    # Per-source confidence table
    if results:
        table = Table(
            title="📌 Sources & Confidence",
            box=box.ROUNDED,
            show_lines=True,
            style="dim"
        )
        table.add_column("File",       style="cyan",  no_wrap=True)
        table.add_column("Page",       justify="center")
        table.add_column("Confidence", justify="center", no_wrap=True)
        table.add_column("Level",      justify="center")
        table.add_column("Preview",    style="white")

        seen = set()
        for r in results:
            doc   = r["doc"]
            src   = os.path.basename(doc.metadata.get("source", "Unknown"))
            page  = str(doc.metadata.get("page", "?"))
            key   = f"{src}-{page}"
            if key not in seen:
                seen.add(key)
                conf  = r["confidence"]
                lbl   = r["label"]
                color = {"High": "green", "Medium": "yellow",
                         "Low": "dark_orange", "Very Low": "red"}.get(lbl, "white")
                preview = doc.page_content[:70].replace("\n", " ") + "..."
                table.add_row(
                    src,
                    page,
                    f"[{color}]{conf:.1f}%[/{color}]",
                    f"[{color}]{lbl}[/{color}]",
                    preview
                )

        console.print(table)

    elif source_docs:
        # Fallback if results not passed
        table = Table(title="📌 Sources", box=box.ROUNDED, show_lines=True, style="dim")
        table.add_column("File",    style="cyan",  no_wrap=True)
        table.add_column("Page",    justify="center")
        table.add_column("Preview", style="white")
        seen = set()
        for doc in source_docs:
            src   = os.path.basename(doc.metadata.get("source", "Unknown"))
            page  = str(doc.metadata.get("page", "?"))
            key   = f"{src}-{page}"
            if key not in seen:
                seen.add(key)
                preview = doc.page_content[:80].replace("\n", " ") + "..."
                table.add_row(src, page, preview)
        console.print(table)

    # Images
    if images:
        display_images_terminal(images)
