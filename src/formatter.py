import os
import re
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

console = Console()

def display_images_terminal(images: list):
    """
    Display image info in terminal.
    Terminal can't show actual images but shows path + info
    so user can open them.
    """
    if not images:
        return

    console.print("\n[bold magenta]🖼  Relevant Diagrams/Figures:[/bold magenta]")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("", style="magenta", width=3)
    table.add_column("File",    style="white")
    table.add_column("Page",    style="dim",    justify="center")
    table.add_column("Size",    style="dim",    justify="right")

    for i, img in enumerate(images, 1):
        size_kb = img["size_bytes"] // 1024
        table.add_row(
            f"[{i}]",
            img["filename"],
            f"p.{img['page']}",
            f"{size_kb} KB"
        )

    console.print(table)
    console.print(
        f"[dim]Images saved in: image_store/ folder — "
        f"open them to view diagrams[/dim]"
    )

def format_answer(
    question:    str,
    answer:      str,
    source_docs: list,
    images:      list = None
):
    """Render the answer in a clean, readable format."""

    # Strip DeepSeek's internal <think>...</think> reasoning tags
    clean_answer = re.sub(
        r"<think>.*?</think>", "", answer, flags=re.DOTALL
    ).strip()

    # Question
    console.print(f"\n[bold cyan]❓ Question:[/bold cyan] {question}\n")

    # Answer panel
    console.print(Panel(
        Markdown(clean_answer),
        title="[bold green]📖 Answer[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))

    # Relevant images
    if images:
        display_images_terminal(images)

    # Source table
    if source_docs:
        table = Table(
            title="📌 Sources",
            box=box.ROUNDED,
            show_lines=True,
            style="dim"
        )
        table.add_column("File",    style="cyan",  no_wrap=True)
        table.add_column("Page",    justify="center")
        table.add_column("Preview", style="white")

        seen = set()
        for doc in source_docs:
            src     = os.path.basename(doc.metadata.get("source", "Unknown"))
            page    = str(doc.metadata.get("page", "?"))
            key     = f"{src}-{page}"
            if key not in seen:
                seen.add(key)
                preview = doc.page_content[:80].replace("\n", " ") + "..."
                table.add_row(src, page, preview)

        console.print(table)