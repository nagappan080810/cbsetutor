import sys
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

def main():
    console.print(Panel.fit(
        "[bold green]🎓 CBSE RAG System[/bold green]\n"
        "[dim]Classes 8–10 | All Subjects | Multilingual[/dim]\n"
        "[dim]Powered by Groq + Ollama + Re-ranking[/dim]",
        border_style="blue"
    ))

    if "--ingest" in sys.argv:
        from src.ingest import ingest_documents, show_tracker_status

        if "--status" in sys.argv:
            # Show what's already ingested
            show_tracker_status()

        elif "--adhoc" in sys.argv:
            # Interactive: pick class → subject → file
            force = "--force" in sys.argv
            ingest_documents(force=force, adhoc=True)

        else:
            # Normal: ingest all new/changed
            force = "--force" in sys.argv
            ingest_documents(force=force, adhoc=False)

        return
    
    if not os.path.exists(os.getenv("CHROMA_DIR", "./chroma_db")):
        console.print("[red]Run ingestion first: python main.py --ingest[/red]")
        return

    from src.selector import (
        select_class, select_subject,
        get_language_instruction, show_selection_summary
    )
    from src.rag_chain import CBSERagChain

    chain = CBSERagChain()

    while True:
        # Select class and subject before questions
        class_name = select_class()
        subject    = select_subject()
        lang_instr = get_language_instruction(subject)

        show_selection_summary(class_name, subject)

        console.print(
            "\n[dim]Ask questions. "
            "Type 'switch' to change class/subject, 'exit' to quit.[/dim]\n"
        )

        # Inner loop — keep asking for same class/subject
        while True:
            try:
                question = input("📝 Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Goodbye![/yellow]")
                return

            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                console.print("[yellow]Goodbye![/yellow]")
                return
            if question.lower() == "switch":
                console.print("\n[cyan]Switching class/subject...[/cyan]")
                break   # go back to outer loop

            chain.ask(question, class_name, subject, lang_instr)

if __name__ == "__main__":
    main()