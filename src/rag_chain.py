import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.retriever import CBSERetriever
from src.formatter import format_answer
from rich.console import Console

load_dotenv()
console = Console()

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "class_name", "subject", "language_instruction"],
    template="""You are an expert CBSE tutor for {class_name}, subject: {subject}.

{language_instruction}

Use ONLY the context below from NCERT textbooks to answer.
If the answer is not in the context, say so clearly in the same language.

Rules:
- For maths: show every step numbered (Step 1, Step 2...)
- For languages (Hindi/Kannada/Tamil/Sanskrit): explain grammar rules clearly
- Define formulas or terms before using them
- Use simple language for a school student
- End with a short summary or key takeaway
- Use markdown formatting (bold for key terms, code blocks for formulas/equations)

---
CONTEXT:
{context}

---
QUESTION: {question}

ANSWER:"""
)

class CBSERagChain:
    def __init__(self):
        self.retriever = CBSERetriever()
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=2048,
        )

    def ask(
        self,
        question:             str,
        class_name:           str,
        subject:              str,
        language_instruction: str
    ):
        console.print(
            f"\n[bold yellow]🔍 Searching {class_name} → {subject}...[/bold yellow]"
        )

        # retrieve_and_rerank now returns list of dicts with doc + confidence
        results = self.retriever.retrieve_and_rerank(
            question,
            class_filter=class_name,
            subject_filter=subject
        )

        if not results:
            console.print(
                "[red]No content found for this class/subject. "
                "Make sure PDFs are placed in the correct folder and ingested.[/red]"
            )
            return

        # Extract doc objects for context building
        top_docs = self.retriever.get_top_docs(results)

        # Compute overall confidence
        overall = self.retriever.overall_confidence(results)

        # Limit context size
        context_parts = []
        total_len     = 0
        for doc in top_docs:
            if total_len + len(doc.page_content) > 2000:
                break
            context_parts.append(doc.page_content)
            total_len += len(doc.page_content)

        context = "\n\n---\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            class_name=class_name.replace("_", " ").title(),
            subject=subject.replace("_", " ").title(),
            language_instruction=language_instruction
        )

        console.print("[bold yellow]🤖 Generating answer...[/bold yellow]\n")

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer   = response.content
        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/red]")
            return

        # Find relevant images
        images = []
        try:
            from src.image_store import find_relevant_images
            images = find_relevant_images(
                class_name=class_name,
                subject=subject,
                source_docs=top_docs,
                max_images=3
            )
        except Exception:
            pass

        format_answer(
            question=question,
            answer=answer,
            source_docs=top_docs,
            images=images,
            results=results,
            overall_confidence=overall
        )
