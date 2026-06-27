Here's a complete, production-ready setup guide:

---

## 🐍 CBSE 10th RAG — Full Python Project Setup

---

## Step 1 — Project Structure

```
cbse10-rag/
├── data/                  ← NCERT PDFs go here
├── chroma_db/             ← Auto-created vector store
├── src/
│   ├── ingest.py          ← Load, chunk & embed PDFs
│   ├── retriever.py       ← Search + re-ranking logic
│   ├── rag_chain.py       ← RAG pipeline with DeepSeek
│   └── formatter.py       ← Clean, readable output
├── main.py                ← Entry point
├── requirements.txt
└── .env
```

Create it:
```bash
mkdir cbse10-rag && cd cbse10-rag
mkdir data src
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

---

## Step 2 — Install Dependencies

Create `requirements.txt`:
```txt
langchain
langchain-community
langchain-ollama
chromadb
pypdf
sentence-transformers
ollama
python-dotenv
rich
numpy
```

Install:
```bash
pip install -r requirements.txt
```

---

## Step 3 — Pull DeepSeek Model via Ollama

```bash
# Start Ollama server (keep this running in a separate terminal)
ollama serve

# Pull DeepSeek R1 (7B — good balance of speed & quality)
ollama pull deepseek-r1:7b

# Pull embedding model
ollama pull nomic-embed-text
```

Verify:
```bash
ollama list
# Should show: deepseek-r1:7b and nomic-embed-text
```

---

## Step 4 — `.env` File

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=deepseek-r1:7b
EMBED_MODEL=nomic-embed-text
CHROMA_DIR=./chroma_db
DATA_DIR=./data
CHUNK_SIZE=600
CHUNK_OVERLAP=80
TOP_K_RETRIEVE=10
TOP_K_RERANK=4
```

---

## Step 5 — `src/ingest.py` (Load & Embed PDFs)

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()

def ingest_documents():
    data_dir   = os.getenv("DATA_DIR", "./data")
    chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
    chunk_size  = int(os.getenv("CHUNK_SIZE", 600))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 80))

    console.print("\n[bold blue]📚 Loading CBSE PDFs...[/bold blue]")
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    console.print(f"   [green]✓ Loaded {len(documents)} pages[/green]")

    # Split — smaller chunks = more precise maths retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_documents(documents)
    console.print(f"   [green]✓ Created {len(chunks)} chunks[/green]")

    console.print("\n[bold blue]🔢 Generating embeddings...[/bold blue]")
    embeddings = OllamaEmbeddings(
        model=embed_model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir
    )
    vectorstore.persist()
    console.print(f"\n[bold green]✅ Vector store saved to {chroma_dir}[/bold green]")
    return vectorstore

if __name__ == "__main__":
    ingest_documents()
```

---

## Step 6 — `src/retriever.py` (Search + Re-Ranking)

This is the key file — re-ranking ensures maths questions get the most relevant chunks, not just semantically similar ones.

```python
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from rich.console import Console

load_dotenv()
console = Console()

# CrossEncoder re-ranker — scores each chunk against the question
# This is much more accurate than vector similarity alone
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CBSERetriever:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.vectorstore = Chroma(
            persist_directory=os.getenv("CHROMA_DIR", "./chroma_db"),
            embedding_function=self.embeddings
        )
        console.print("[dim]Loading re-ranker model...[/dim]")
        self.reranker = CrossEncoder(RERANKER_MODEL)
        self.top_k_retrieve = int(os.getenv("TOP_K_RETRIEVE", 10))
        self.top_k_rerank   = int(os.getenv("TOP_K_RERANK", 4))

    def retrieve_and_rerank(self, question: str) -> list:
        # Step 1: Broad vector search — get top 10 candidates
        candidates = self.vectorstore.similarity_search(
            question, k=self.top_k_retrieve
        )

        if not candidates:
            return []

        # Step 2: Re-rank using CrossEncoder
        # CrossEncoder reads question + chunk together for deeper relevance scoring
        pairs = [[question, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        # Step 3: Sort by score, keep top K
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        top_docs = [doc for _, doc in ranked[:self.top_k_rerank]]

        console.print(
            f"[dim]  Retrieved {len(candidates)} → re-ranked → top {len(top_docs)}[/dim]"
        )
        return top_docs
```

---

## Step 7 — `src/formatter.py` (Readable Output)

```python
import re
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

console = Console()

def format_answer(question: str, answer: str, source_docs: list):
    """Render the answer in a clean, readable format."""

    # Strip DeepSeek's internal <think>...</think> reasoning tags
    clean_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    # Print the question
    console.print(f"\n[bold cyan]❓ Question:[/bold cyan] {question}\n")

    # Print answer inside a styled panel
    console.print(Panel(
        Markdown(clean_answer),
        title="[bold green]📖 Answer[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))

    # Source table
    if source_docs:
        table = Table(
            title="📌 Sources",
            box=box.ROUNDED,
            show_lines=True,
            style="dim"
        )
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Page", justify="center")
        table.add_column("Preview", style="white")

        seen = set()
        for doc in source_docs:
            src  = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = str(doc.metadata.get("page", "?"))
            key  = f"{src}-{page}"
            if key not in seen:
                seen.add(key)
                preview = doc.page_content[:80].replace("\n", " ") + "..."
                table.add_row(src, page, preview)

        console.print(table)

import os  # add at top if missing
```

---

## Step 8 — `src/rag_chain.py` (Full RAG Pipeline)

```python
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from src.retriever import CBSERetriever
from src.formatter import format_answer
from rich.console import Console

load_dotenv()
console = Console()

# Maths-aware prompt — instructs DeepSeek to show steps clearly
MATH_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert CBSE Class 10 tutor specialising in Mathematics and Science.

Use ONLY the context below (from NCERT textbooks) to answer the question.
If the answer is not in the context, say: "This topic is not found in the loaded textbooks."

Rules for your answer:
- For maths problems: show every step clearly, numbered (Step 1, Step 2...)
- Define formulas before using them
- Use simple language suitable for a Class 10 student
- End with a short summary or key takeaway
- Format using markdown (bold for key terms, code blocks for formulas)

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
        self.llm = OllamaLLM(
            model=os.getenv("LLM_MODEL", "deepseek-r1:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,       # Low temp = factual, consistent answers
            num_ctx=4096           # Context window
        )

    def ask(self, question: str):
        console.print(f"\n[bold yellow]🔍 Searching CBSE textbooks...[/bold yellow]")

        # Retrieve + re-rank
        top_docs = self.retriever.retrieve_and_rerank(question)

        if not top_docs:
            console.print("[red]No relevant content found. Make sure PDFs are ingested.[/red]")
            return

        # Build context string from top chunks
        context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])

        # Build final prompt
        prompt = MATH_PROMPT.format(context=context, question=question)

        console.print("[bold yellow]🤖 DeepSeek is thinking...[/bold yellow]\n")

        # Stream the response
        answer = ""
        for chunk in self.llm.stream(prompt):
            answer += chunk

        # Format and display
        format_answer(question, answer, top_docs)
```

---

## Step 9 — `main.py` (Entry Point)

```python
import sys
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

def main():
    console.print(Panel.fit(
        "[bold green]🎓 CBSE Class 10 RAG System[/bold green]\n"
        "[dim]Powered by DeepSeek R1 + Ollama + Re-ranking[/dim]",
        border_style="blue"
    ))

    # If --ingest flag passed, run ingestion first
    if "--ingest" in sys.argv:
        from src.ingest import ingest_documents
        ingest_documents()
        console.print("\n[green]Ingestion complete. Restart without --ingest to query.[/green]")
        return

    # Check chroma_db exists
    if not os.path.exists(os.getenv("CHROMA_DIR", "./chroma_db")):
        console.print("[red]Vector store not found! Run: python main.py --ingest[/red]")
        return

    from src.rag_chain import CBSERagChain
    chain = CBSERagChain()

    console.print("\n[dim]Type your question below. Commands: 'exit' to quit[/dim]\n")

    while True:
        try:
            question = input("📝 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        chain.ask(question)

if __name__ == "__main__":
    main()
```

---

## Step 10 — Run It

```bash
# 1. Place your NCERT PDFs in the data/ folder

# 2. Ingest documents (run once)
python main.py --ingest

# 3. Start asking questions
python main.py
```

---

## 💬 Example Output

```
📝 Your question: How do you find the roots of a quadratic equation?

🔍 Searching CBSE textbooks...
  Retrieved 10 → re-ranked → top 4

🤖 DeepSeek is thinking...

╭─────────────────── 📖 Answer ────────────────────╮
│                                                   │
│  The roots of a quadratic equation ax² + bx + c  │
│  = 0 can be found using:                         │
│                                                   │
│  **Quadratic Formula:**                           │
│  `x = (-b ± √(b² - 4ac)) / 2a`                  │
│                                                   │
│  **Step 1:** Identify a, b, and c                │
│  **Step 2:** Calculate the discriminant D = b²-4ac│
│  **Step 3:** If D ≥ 0, roots are real            │
│  **Step 4:** Apply the formula                   │
│                                                   │
│  **Key Takeaway:** The discriminant tells you    │
│  how many real roots exist.                      │
╰───────────────────────────────────────────────────╯

📌 Sources
┌─────────────────┬──────┬──────────────────────────┐
│ File            │ Page │ Preview                  │
├─────────────────┼──────┼──────────────────────────┤
│ mathematics.pdf │  23  │ The standard form of a.. │
│ mathematics.pdf │  25  │ Discriminant D = b²-4ac..│
└─────────────────┴──────┴──────────────────────────┘
```

---

## 🔑 Key Design Decisions

| Choice | Why |
|---|---|
| **DeepSeek R1:7b** | Best reasoning for maths step-by-step explanations |
| **CrossEncoder re-ranker** | Far more accurate than vector similarity alone for precise maths queries |
| **Chunk size 600** | Small enough to isolate specific formulas and theorems |
| **Temperature 0.1** | Keeps answers factual, not creative |
| **Rich library** | Clean terminal formatting — panels, tables, markdown rendering |

Let me know if you want me to add a **Streamlit web UI** on top of this!


# Ingest all new PDFs
python main.py --ingest

# Re-ingest everything from scratch
python main.py --ingest --force

# Pick specific class → subject → file(s) to fix
python main.py --ingest --adhoc

# Force re-ingest a specific file (adhoc + force)
python main.py --ingest --adhoc --force

# See what's already been ingested
python main.py --ingest --status

images are not showing properly..

if failed, it need to maintain it as failed, even if chunks is zero to be treated as failed 

if we give adhoc, only that subject files will be reloaded even if the ingest json file is not there

whether ingest json file will get updated even if we do multiple runs?

We need to make this as rest api system also for the workflows like passing the questions and getting the answers, we can pass standard and subject also in each question.
Answer will be generated as async and it will be passed as server sent events so the client receives token by token and render it. 

We need apis to fetch the images after we get those image titles in the answer.. 
lets make fast api to get response

/opt/projects/cbsetutor/src/ingest.py:699: LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
  vectorstore.persist()

/opt/projects/cbsetutor/src/retriever.py:36: LangChainDeprecationWarning: The class `OllamaEmbeddings` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the `langchain-ollama package and should be used instead. To use it run `pip install -U `langchain-ollama` and import as `from `langchain_ollama import OllamaEmbeddings``.
  return OllamaEmbeddings(
/opt/projects/cbsetutor/src/retriever.py:15: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the `langchain-chroma package and should be used instead. To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma``.
  self.vectorstore = Chroma(
Loading re-ranker model...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

dont hallcuniate the answers, answers need to be accurate. 

We can have coordinator which delegates rag agent to find and rank the answers and then the teacher agent verifies the answer, if it is not valid, ask web agent to search the answer and then teacher verifies the answer, if it is still not valid it can say that not able to find the answer. 

[Student Query] 
       │
       ▼
┌──────────────┐
│ Coordinator  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  RAG Agent   │ ──► Extracts facts from NCERT Vector DB
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Teacher Agent │ ──► Checks against CBSE marking rubrics
└──────┬───────┘
       ├───────────────────────┐
       ▼ (If Valid)            ▼ (If Invalid / Incomplete)
[Deliver Answer]        ┌──────────────┐
                        │  Web Agent   │ ──► Searches trusted CBSE sites
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Teacher Agent │ ──► Final validation check
                        └──────┬───────┘
                               ├──────────────────────┐
                               ▼ (If Valid)           ▼ (If Still Invalid)
                        [Deliver Answer]       [Polite "Not Found" Msg]