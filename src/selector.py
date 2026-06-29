import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

CLASSES = {
    "1": "class_8",
    "2": "class_9",
    "3": "class_10"
}

SUBJECTS = {
    "1":  "mathematics",
    "2":  "science",
    "3":  "socialscience",
    "4":  "english",
    "5":  "hindi",
    "6":  "kannada",
    "7":  "tamil",
    "8":  "sanskrit"
}

LANGUAGE_PROMPTS = {
    "english":      "Respond in English.",
    "hindi":        "हिंदी में जवाब दें। (Respond in Hindi)",
    "kannada":      "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ. (Respond in Kannada)",
    "tamil":        "தமிழில் பதில் சொல்லுங்கள். (Respond in Tamil)",
    "sanskrit":     "संस्कृते उत्तरं देहि। (Respond in Sanskrit)",
    "mathematics":  "Respond in English with clear step-by-step solutions.",
    "science":      "Respond in English with clear explanations.",
    "socialscience": "Respond in English.",
}

def select_class() -> str:
    """Show class selection menu and return selected class."""
    console.print("\n[bold cyan]📚 Select Class:[/bold cyan]")

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("No.", style="yellow", width=5)
    table.add_column("Class", style="white")

    for key, val in CLASSES.items():
        display = val.replace("_", " ").title()
        table.add_row(f"[{key}]", display)

    console.print(table)

    while True:
        choice = input("Enter number (1-3): ").strip()
        if choice in CLASSES:
            selected = CLASSES[choice]
            console.print(
                f"[green]✓ Selected: {selected.replace('_', ' ').title()}[/green]"
            )
            return selected
        console.print("[red]Invalid choice. Enter 1, 2, or 3.[/red]")

def select_subject() -> str:
    """Show subject selection menu and return selected subject."""
    console.print("\n[bold cyan]📖 Select Subject:[/bold cyan]")

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("No.", style="yellow", width=5)
    table.add_column("Subject", style="white")

    for key, val in SUBJECTS.items():
        display = val.replace("_", " ").title()
        table.add_row(f"[{key}]", display)

    console.print(table)

    while True:
        choice = input("Enter number (1-8): ").strip()
        if choice in SUBJECTS:
            selected = SUBJECTS[choice]
            console.print(
                f"[green]✓ Selected: {selected.replace('_', ' ').title()}[/green]"
            )
            return selected
        console.print("[red]Invalid choice. Enter 1-8.[/red]")

def get_language_instruction(subject: str) -> str:
    """Return language instruction based on subject."""
    return LANGUAGE_PROMPTS.get(subject, "Respond in English.")

def show_selection_summary(class_name: str, subject: str):
    """Show what the user selected."""
    console.print(Panel(
        f"[bold]Class:[/bold]   {class_name.replace('_', ' ').title()}\n"
        f"[bold]Subject:[/bold] {subject.replace('_', ' ').title()}",
        title="[bold green]✅ Session Started[/bold green]",
        border_style="green",
        padding=(0, 2)
    ))