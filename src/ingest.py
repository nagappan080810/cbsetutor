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
    "mathematics", "science", "socialscience",
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

# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC METADATA ENRICHMENT
# Every chunk in ChromaDB gets rich metadata so retrieval can filter by
# content type, chapter, section, difficulty signal, and more.
#
# Metadata fields stored per chunk:
#   source          – PDF filename
#   class           – e.g. "class_10"
#   subject         – e.g. "science"
#   page            – 0-based page number
#   chapter         – "Chapter 1 – Chemical Reactions"
#   chapter_num     – 1  (int, for ordering/filtering)
#   section         – "1.2 Types of Chemical Reactions"
#   subsection      – "1.2.1 Combination Reactions"
#   content_type    – one of:
#                       "question"      – exercise / textbook question
#                       "answer"        – worked example or solved answer
#                       "definition"    – "X is defined as …"
#                       "fact"          – declarative factual statement
#                       "summary"       – chapter summary / key points
#                       "example"       – "Example:" block
#                       "formula"       – contains mathematical formula
#                       "table"         – tabular data
#                       "figure_ref"    – refers to a diagram/figure
#                       "activity"      – lab / activity / do-it-yourself
#                       "note"          – callout box / think & discuss
#                       "exercise"      – end-of-chapter exercises
#                       "introduction"  – chapter/section intro text
#                       "body"          – general body text (fallback)
#   has_formula     – bool: chunk contains a math/chemical formula
#   has_table       – bool: chunk appears to contain tabular data
#   has_figure_ref  – bool: chunk references a Figure/Diagram
#   keyword_hints   – comma-separated top-5 content words (for debug/search)
#   bloom_level     – estimated Bloom's level: "remember","understand",
#                     "apply","analyse","evaluate" (helps difficulty routing)
#   word_count      – word count of the chunk
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re
import math as _math
from collections import Counter as _Counter

# ── Pattern library ───────────────────────────────────────────────────────────

# Chapter heading: "Chapter 1", "CHAPTER 3 – Motion", "1. Motion"
_RE_CHAPTER = _re.compile(
    r"^(?:chapter\s+(\d{1,2})[\s:–\-]*(.*?)|(\d{1,2})\.\s+([A-Z][A-Za-z ,\-&'/]+))$",
    _re.IGNORECASE | _re.MULTILINE,
)
# Section: "1.1 Introduction", "2.3 Types of Motion"
_RE_SECTION = _re.compile(
    r"^(\d{1,2})\.(\d{1,2})\s+([A-Z][A-Za-z ,\-&'()/]+)$",
    _re.MULTILINE,
)
# Subsection: "1.2.1 Combination Reactions"
_RE_SUBSECTION = _re.compile(
    r"^(\d{1,2}\.\d{1,2}\.\d{1,2})\s+([A-Z][A-Za-z ,\-&'()/]+)$",
    _re.MULTILINE,
)

# Content-type signal patterns
_RE_QUESTION    = _re.compile(
    r"(?:^|\n)\s*(?:Q\.?\s*\d+|question\s+\d+|\d+\.\s+(?:what|why|how|when|where|who|which|define|explain|describe|state|list|give|find|calculate|solve|show|prove|draw|name|differentiate|compare|discuss|write))",
    _re.IGNORECASE,
)
_RE_THINK_Q     = _re.compile(
    r"(?:think\s+and\s+(?:discuss|answer)|in-text\s+question|activity\s+\d+|try\s+this|do\s+you\s+know\?)",
    _re.IGNORECASE,
)
_RE_ANSWER      = _re.compile(
    r"(?:^|\n)\s*(?:solution|ans(?:wer)?|sol\.)\s*[:\-]",
    _re.IGNORECASE,
)
_RE_EXAMPLE     = _re.compile(
    r"(?:^|\n)\s*example\s*\d*\s*[:\-]?",
    _re.IGNORECASE,
)
_RE_DEFINITION  = _re.compile(
    r"(?:is\s+defined\s+as|is\s+called|are\s+known\s+as|refers?\s+to|means?\s+that|is\s+a\s+(?:process|type|form|kind|method|substance|property|phenomenon))",
    _re.IGNORECASE,
)
_RE_SUMMARY     = _re.compile(
    r"(?:^|\n)\s*(?:what\s+you\s+have\s+learnt|key\s+(?:points?|takeaways?|terms?|concepts?)|summary|in\s+this\s+chapter|points?\s+to\s+remember|recap)",
    _re.IGNORECASE,
)
_RE_EXERCISE    = _re.compile(
    r"(?:^|\n)\s*(?:exercises?|problems?|assignments?|practice\s+questions?|additional\s+questions?|ncert\s+solutions?)\s*\n",
    _re.IGNORECASE,
)
_RE_ACTIVITY    = _re.compile(
    r"(?:^|\n)\s*(?:activity\s+\d+|lab\s+activity|experiment|practical|let\s+us\s+(?:do|try|find)|hands[\-\s]on)",
    _re.IGNORECASE,
)
_RE_NOTE        = _re.compile(
    r"(?:^|\n)\s*(?:note\s*:|remember\s*:|caution\s*:|important\s*:|did\s+you\s+know\??|fun\s+fact|think\s*:)",
    _re.IGNORECASE,
)
_RE_FORMULA     = _re.compile(
    r"(?:[A-Za-z]\s*=\s*[\w\(\)\+\-\*/\^]+|"       # algebra: v = u + at
    r"[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+|"           # chemical: H2SO4, CO2, NaCl
    r"\\frac|\\sqrt|∫|∑|∆|α|β|γ|λ|μ|σ|ω|"          # LaTeX / Greek
    r"\d+\s*[×x]\s*\d+|"                             # multiplication
    r"(?:mol|kg|kJ|kPa|atm|°C|°F|Hz|N\/m))",        # units
    _re.IGNORECASE,
)
_RE_TABLE       = _re.compile(
    r"(?:\|\s*[-:]+\s*\||\t[^\t]+\t[^\t]+\t|"       # markdown or tab table
    r"(?:s\.?\s*no\.?|sl\.?\s*no\.?)\s*[\.\|:])",    # "S.No." header
    _re.IGNORECASE,
)
_RE_FIGURE      = _re.compile(
    r"(?:fig(?:ure)?\.?\s*\d+|diagram\s+\d+|see\s+fig(?:ure)?|as\s+shown\s+in|refer\s+to\s+fig(?:ure)?)",
    _re.IGNORECASE,
)
_RE_INTRO       = _re.compile(
    r"(?:^|\n)\s*(?:introduction|overview|in\s+this\s+chapter\s+we\s+(?:will|shall)|let\s+us\s+(?:begin|start|recall|revise))",
    _re.IGNORECASE,
)

# ── Subject-domain classifiers ────────────────────────────────────────────────

# Science sub-domains
_RE_PHYSICS = _re.compile(
    r"\b(?:force|motion|velocity|acceleration|momentum|energy|work|power|"
    r"light|sound|wave|electricity|current|voltage|resistance|magnetic|"
    r"gravitation|pressure|floatation|newton|ohm|joule|watt|reflection|"
    r"refraction|lens|mirror|circuit|charge|potential|capacitor|induction|"
    r"oscillation|frequency|amplitude|thermodynamics|heat|temperature)\b",
    _re.IGNORECASE,
)
_RE_CHEMISTRY = _re.compile(
    r"\b(?:element|compound|mixture|acid|base|salt|reaction|equation|"
    r"oxidation|reduction|metal|non.?metal|ion|bond|covalent|ionic|"
    r"periodic|carbon|organic|hydrocarbon|polymer|pH|mole|mol|atom|"
    r"molecule|chemical|solution|concentration|electrolysis|catalyst|"
    r"corrosion|combustion|hydrogen|oxygen|nitrogen|chlorine|sodium|"
    r"calcium|iron|copper|zinc|sulphate|carbonate|oxide)\b",
    _re.IGNORECASE,
)
_RE_BIOLOGY = _re.compile(
    r"\b(?:cell|tissue|organ|organism|photosynthesis|respiration|"
    r"reproduction|heredity|evolution|ecosystem|DNA|chromosome|gene|"
    r"protein|enzyme|hormone|nerve|brain|heart|blood|lungs|kidney|"
    r"digestion|nutrition|excretion|plant|animal|bacteria|virus|fungi|"
    r"microorganism|classification|species|adaptation|food.?chain|"
    r"biodiversity|pollination|germination|osmosis|diffusion)\b",
    _re.IGNORECASE,
)

# Social Science sub-domains
_RE_HISTORY = _re.compile(
    r"\b(?:war|revolution|empire|century|colonial|independence|treaty|"
    r"civilisation|civilization|dynasty|nationalist|movement|rebellion|"
    r"partition|unification|imperialism|feudalism|renaissance|reformation|"
    r"ancient|medieval|modern|historical|king|queen|ruler|battle|"
    r"trade\s+route|colonialism|peasant|uprising|constitution\s+of\s+india|"
    r"freedom\s+fighter|British\s+raj|mughal|maurya|gupta|delhi\s+sultanate)\b",
    _re.IGNORECASE,
)
_RE_GEOGRAPHY = _re.compile(
    r"\b(?:climate|rainfall|plateau|river|latitude|longitude|vegetation|"
    r"soil|region|mineral|resource|mountain|plain|delta|glacier|erosion|"
    r"deposition|atmosphere|pressure|wind|monsoon|drought|flood|map|"
    r"topography|contour|scale|irrigation|agriculture|crop|forest|"
    r"population|urbanization|migration|transport|communication|"
    r"ocean|sea|coast|peninsula|island|watershed|tributary)\b",
    _re.IGNORECASE,
)
_RE_CIVICS = _re.compile(
    r"\b(?:constitution|parliament|rights|democracy|election|government|"
    r"court|fundamental|directive|federalism|judiciary|legislature|"
    r"executive|citizenship|sovereignty|secularism|republic|amendment|"
    r"president|prime\s+minister|cabinet|lok\s+sabha|rajya\s+sabha|"
    r"panchayat|municipality|political\s+party|suffrage|representation|"
    r"equality|justice|liberty|fraternity|secularism)\b",
    _re.IGNORECASE,
)
_RE_ECONOMICS = _re.compile(
    r"\b(?:GDP|poverty|market|demand|supply|money|bank|sector|employment|"
    r"consumer|price|inflation|income|wage|tax|subsidy|budget|trade|"
    r"export|import|globalization|liberalization|privatization|"
    r"development|growth|per\s+capita|inequality|rural|urban|"
    r"industry|service|agriculture\s+sector|credit|loan|interest)\b",
    _re.IGNORECASE,
)

# Question location — in-text (mid-chapter) vs end-of-chapter exercise
_RE_INTEXT_Q = _re.compile(
    r"(?:^|\n)\s*(?:think\s+and\s+(?:discuss|answer)|in.?text\s+question|"
    r"do\s+you\s+know\??|try\s+this|can\s+you\s+(?:tell|think|find|answer)|"
    r"discuss\s+with\s+your\s+(?:teacher|friends?|classmates?)|"
    r"activity\s+\d+|checkpoint\s*\d*|quick\s+check|pause\s+and\s+think|"
    r"बूझो तो जानें|क्या आप जानते हैं)",
    _re.IGNORECASE | _re.MULTILINE,
)
_RE_EXERCISE_Q = _re.compile(
    r"(?:^|\n)\s*(?:exercises?\s*\n|exercise\s+\d|end[\s\-]of[\s\-]chapter|"
    r"chapter\s+end|practice\s+questions?\s*\n|additional\s+(?:exercises?|questions?)|"
    r"questions?\s+and\s+answers?\s*\n|ncert\s+(?:exercises?|solutions?)|"
    r"textbook\s+(?:exercises?|questions?)|अभ्यास\s*\n|प्रश्न\s+अभ्यास)",
    _re.IGNORECASE | _re.MULTILINE,
)

# Requires diagram
_RE_NEEDS_DIAGRAM = _re.compile(
    r"\b(?:draw\s+(?:a|the|an)\s+(?:labelled?\s+)?(?:diagram|figure|sketch|graph|circuit|ray\s+diagram)|"
    r"sketch\s+(?:a|the)|"
    r"label\s+the\s+(?:diagram|figure|parts?)|"
    r"show\s+(?:with\s+(?:a|the)\s+help\s+of\s+(?:a\s+)?diagram|diagrammatically)|"
    r"illustrate\s+with\s+(?:a\s+)?diagram|"
    r"construct\s+(?:a|the)\s+(?:triangle|angle|circle|quadrilateral)|"
    r"plot\s+(?:a|the)\s+(?:graph|curve)|"
    r"mark\s+(?:on\s+(?:a\s+)?(?:map|diagram|figure)))\b",
    _re.IGNORECASE,
)


def _classify_science_domain(text: str) -> str:
    """Return physics/chemistry/biology for science subject chunks."""
    p = len(_RE_PHYSICS.findall(text))
    c = len(_RE_CHEMISTRY.findall(text))
    b = len(_RE_BIOLOGY.findall(text))
    if p == 0 and c == 0 and b == 0:
        return "general"
    best = max(("physics", p), ("chemistry", c), ("biology", b), key=lambda x: x[1])
    return best[0]


def _classify_social_domain(text: str) -> str:
    """Return history/geography/civics/economics for social science chunks."""
    h = len(_RE_HISTORY.findall(text))
    g = len(_RE_GEOGRAPHY.findall(text))
    c = len(_RE_CIVICS.findall(text))
    e = len(_RE_ECONOMICS.findall(text))
    if h == 0 and g == 0 and c == 0 and e == 0:
        return "general"
    best = max(("history", h), ("geography", g), ("civics", c), ("economics", e), key=lambda x: x[1])
    return best[0]


def _classify_question_location(text: str) -> str:
    """Distinguish intext questions from end-of-chapter exercises."""
    if _RE_EXERCISE_Q.search(text):
        return "exercise"
    if _RE_INTEXT_Q.search(text):
        return "intext"
    return "body"


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE SUBJECT ENRICHMENT
# Applied only when subject in {english, hindi, kannada, tamil, sanskrit}.
# Adds two extra metadata fields:
#   lang_sub_type   – fine-grained content category (see values below)
#   sentence_class  – "short_sentence"|"long_sentence"|"passage"|"word_list"
#
# lang_sub_type values:
#   grammar_rule / grammar_example / grammar_exercise
#   comprehension / comprehension_question
#   short_answer_q / long_answer_q
#   dialogue / letter_writing / essay_writing / story / poem
#   summary_passage / vocabulary / translation
#   note_making / report_writing / speech
#   body  (fallback)
# ═══════════════════════════════════════════════════════════════════════════════

_LANGUAGE_SUBJECTS = {"english", "hindi", "kannada", "tamil", "sanskrit"}

_RE_LANG_GRAMMAR_RULE = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:a|an)\s+(?:noun|verb|adjective|adverb|pronoun|preposition|conjunction|interjection|article|tense|clause|phrase|sentence)\s+is\b|"
    r"rule\s*[:–\-]\s*|"
    r"(?:present|past|future)\s+(?:simple|continuous|perfect|tense)|"
    r"(?:active|passive)\s+voice|"
    r"(?:direct|indirect)\s+(?:speech|narration)|"
    r"(?:singular|plural)\s+(?:form|number)|"
    r"(?:countable|uncountable)\s+noun|"
    r"(?:transitive|intransitive)\s+verb|"
    r"(?:coordinate|subordinate)\s+(?:clause|conjunction)|"
    r"(?:कारक|संधि|समास|वचन|लिंग|काल|विभक्ति|क्रिया|विशेषण|सर्वनाम)|"
    r"(?:ಸಂಧಿ|ಸಮಾಸ|ಕಾರಕ|ಕ್ರಿಯಾ|ವಿಭಕ್ತಿ|ನಾಮಪದ|ಕ್ರಿಯಾಪದ)|"
    r"(?:சந்தி|வேர்ச்சொல்|வினையெச்சம்|பெயரெச்சம்|விகுதி)|"
    r"(?:संधि|समास|कारक|विभक्ति|धातु|प्रत्यय|उपसर्ग)"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_GRAMMAR_EXAMPLE = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"e\.?\s*g\.?\s*[:\.,]|"
    r"for\s+example\s*[:\.,]|"
    r"example\s*[:\-–]\s*[\"']?[A-Z]|"
    r"(?:correct|incorrect)\s*[:\-–]|"
    r"(?:उदाहरण|उदाहरण\s*:|उदा\.)|"
    r"(?:ಉದಾಹರಣೆ|ಉದಾ\.)|"
    r"(?:எடுத்துக்காட்டு|எ\.கா\.)|"
    r"(?:उदाहरण|यथा)"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_GRAMMAR_EXERCISE = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"fill\s+in\s+the\s+(?:blanks?|gaps?)|fill\s+up|"
    r"rewrite\s+the\s+following|change\s+the\s+following|transform\s+the\s+following|"
    r"match\s+the\s+(?:following|columns?|words?)|"
    r"underline\s+the|circle\s+the|identify\s+the\s+(?:noun|verb|adjective|subject|object)|"
    r"make\s+sentences?\s+using|correct\s+the\s+(?:errors?|mistakes?|sentences?)|"
    r"use\s+the\s+following\s+words?\s+in\s+sentences?|"
    r"insert\s+(?:articles?|prepositions?|conjunctions?)|"
    r"do\s+as\s+directed|as\s+directed\s+in\s+brackets?|"
    r"रिक्त\s+स्थान|सही\s+शब्द\s+भरिए|वाक्य\s+बनाइए|"
    r"ಖಾಲಿ\s+ತುಂಬಿರಿ|ವಾಕ್ಯ\s+ರಚಿಸಿ|"
    r"வெற்றிட\s+நிரப்புக|சொற்றொடர்\s+அமை"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_COMPREHENSION = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"read\s+the\s+(?:following\s+)?(?:passage|extract|paragraph|text)\s+(?:carefully\s+)?and\s+(?:answer|do)|"
    r"(?:unseen\s+)?(?:passage\s+for\s+)?comprehension|"
    r"based\s+on\s+(?:the\s+)?(?:above\s+)?(?:passage|extract)|"
    r"गद्यांश|पद्यांश|अपठित\s+गद्यांश|"
    r"ಗद्यभाग|ಪद्यभाग|ಅಪಠಿತ\s+ಗद्यभाग|"
    r"உரைநடை\s+பகுதி|கவிதை\s+பகுதி"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_COMP_QUESTION = _re.compile(
    r"(?:"
    r"answer\s+the\s+following\s+questions?\s+(?:based\s+on|from)\s+the\s+(?:above\s+)?(?:passage|extract)|"
    r"(?:on\s+the\s+basis\s+of|according\s+to)\s+the\s+(?:above\s+)?(?:passage|extract)|"
    r"निम्नलिखित\s+प्रश्नों\s+के\s+उत्तर\s+दीजिए|"
    r"ಕೆಳಗಿನ\s+ಪ್ರಶ್ನೆಗಳಿಗೆ\s+ಉತ್ತರಿಸಿ"
    r")",
    _re.IGNORECASE,
)

_RE_LANG_SHORT_Q = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"answer\s+(?:briefly|in\s+(?:one|two|a\s+few)\s+(?:words?|sentences?|lines?))|"
    r"give\s+(?:a\s+)?short\s+(?:answer|note|description)|"
    r"in\s+(?:not\s+more\s+than|about)\s+(?:\d+|twenty|thirty|fifty)\s+words?|"
    r"name|define|state|mention\s+(?:the|any|two|three|four|five)|"
    r"संक्षेप\s+में|एक\s+शब्द\s+में|एक\s+वाक्य\s+में|"
    r"ಸಂಕ್ಷಿಪ್ತವಾಗಿ|ಒಂದು\s+ವಾಕ್ಯದಲ್ಲಿ"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_LONG_Q = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"write\s+(?:a\s+)?(?:detailed|critical|long|elaborate)\s+(?:note|answer|essay|description|paragraph)|"
    r"discuss\s+(?:in\s+detail|at\s+length|with\s+examples?)|"
    r"describe\s+(?:in\s+detail|at\s+length|elaborately)|"
    r"in\s+(?:not\s+less\s+than|about|at\s+least)\s+(?:\d{2,3}|hundred|two\s+hundred)\s+words?|"
    r"write\s+an?\s+(?:essay|composition|paragraph|article|speech|report|letter)\s+(?:on|about|describing|discussing)|"
    r"विस्तार\s+से\s+लिखिए|निबंध\s+लिखिए|"
    r"ವಿವರವಾಗಿ\s+ಬರೆಯಿರಿ|ಪ್ರಬಂಧ\s+ಬರೆಯಿರಿ"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_DIALOGUE = _re.compile(
    r"(?:"
    r"(?:\b[A-Z][a-z]+\s*:\s+[A-Z\"].{10,}\n){2,}|"
    r"conversation\s+between|dialogue\s+between|talking\s+to\s+each\s+other|"
    r"वार्तालाप|संवाद|"
    r"ಸಂಭಾಷಣೆ|"
    r"உரையாடல்"
    r")",
    _re.IGNORECASE,
)

_RE_LANG_LETTER = _re.compile(
    r"(?:"
    r"(?:formal|informal|friendly)\s+letter|write\s+a\s+letter\s+to|"
    r"dear\s+(?:sir|madam|friend),|"
    r"yours?\s+(?:sincerely|faithfully|truly|affectionately|lovingly)|"
    r"पत्र\s+लेखन|औपचारिक\s+पत्र|अनौपचारिक\s+पत्र|"
    r"ಪತ್ರ\s+ಬರೆಯಿರಿ|ಔಪಚಾರಿಕ\s+ಪತ್ರ"
    r")",
    _re.IGNORECASE,
)

_RE_LANG_ESSAY = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"write\s+an?\s+essay\s+(?:on|about)|essay\s+(?:on|about|writing)|"
    r"composition\s+(?:on|about|writing)|"
    r"निबंध\s+(?:लेखन|लिखिए|लिखो)|"
    r"ಪ್ರಬಂಧ\s+ರಚಿಸಿ"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_STORY = _re.compile(
    r"(?:"
    r"once\s+(?:upon\s+a\s+time|there\s+was)|long\s+ago\s+there|"
    r"story\s+(?:of|about)|write\s+a\s+story|storywriting|"
    r"कहानी\s+लेखन|कहानी\s+लिखिए|एक\s+बार\s+की\s+बात|"
    r"ಕಥೆ\s+ಬರೆಯಿರಿ|ಒಮ್ಮೆ\s+ಒಬ್ಬ"
    r")",
    _re.IGNORECASE,
)

_RE_LANG_POEM = _re.compile(
    r"(?:poem|poetry|rhyme|stanza|verse|couplet|"
    r"कविता|पद्य|दोहा|चौपाई|श्लोक|"
    r"ಕವಿತೆ|ಪದ್ಯ|"
    r"கவிதை|பாடல்|"
    r"श्लोक|पद्य)",
    _re.IGNORECASE,
)

_RE_LANG_VOCABULARY = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"word\s+meanings?|meanings?\s+of\s+(?:the\s+)?(?:following\s+)?words?|"
    r"synonyms?\s+(?:of|for)|antonyms?\s+(?:of|for)|homophones?|homonyms?|"
    r"glossary|vocabulary|word\s+bank|new\s+words?|difficult\s+words?|"
    r"match\s+the\s+words?\s+with\s+their\s+meanings?|"
    r"शब्दार्थ|पर्यायवाची|विलोम\s+शब्द|मुहावरे|लोकोक्तियाँ|"
    r"ಶಬ್ದಾರ್ಥ|ಸಮಾನಾರ್ಥಕ|ವಿರುದ್ಧಾರ್ಥಕ|"
    r"சொற்பொருள்|எதிர்ச்சொல்|ஒத்த\s+சொல்"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_TRANSLATION = _re.compile(
    r"(?:^|\n)\s*(?:"
    r"translate\s+(?:the\s+following|into|from)|translation\s+(?:of|into)|"
    r"अनुवाद\s+(?:कीजिए|करो|लिखिए)|"
    r"ಅನುವಾದ\s+ಮಾಡಿ|ಭಾಷಾಂತರ\s+ಮಾಡಿ|"
    r"மொழிபெயர்|மொழிபெயர்ப்பு"
    r")",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_NOTE_MAKING = _re.compile(
    r"(?:^|\n)\s*(?:note[\s-]?making|note[\s-]?taking|make\s+notes?\s+(?:of|on|from)|"
    r"summarize\s+the\s+(?:following\s+)?(?:passage|text)|summary\s+writing)",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_REPORT = _re.compile(
    r"(?:^|\n)\s*(?:report\s+writing|write\s+a\s+(?:news\s+)?report|newspaper\s+report)",
    _re.IGNORECASE | _re.MULTILINE,
)

_RE_LANG_SPEECH = _re.compile(
    r"(?:^|\n)\s*(?:speech\s+(?:writing|on)|write\s+a\s+speech|debate\s+(?:on|about)|"
    r"speaking\s+(?:activity|practice)|speech\s+by)",
    _re.IGNORECASE | _re.MULTILINE,
)


def _classify_lang_sub_type(text: str) -> str:
    """Fine-grained language content type. Priority: structural > content."""
    t = text.strip()
    if _RE_LANG_GRAMMAR_EXERCISE.search(t): return "grammar_exercise"
    if _RE_LANG_COMP_QUESTION.search(t):    return "comprehension_question"
    if _RE_LANG_LONG_Q.search(t):           return "long_answer_q"
    if _RE_LANG_SHORT_Q.search(t):          return "short_answer_q"
    if _RE_LANG_TRANSLATION.search(t):      return "translation"
    if _RE_LANG_NOTE_MAKING.search(t):      return "note_making"
    if _RE_LANG_REPORT.search(t):           return "report_writing"
    if _RE_LANG_SPEECH.search(t):           return "speech"
    if _RE_LANG_ESSAY.search(t):            return "essay_writing"
    if _RE_LANG_LETTER.search(t):           return "letter_writing"
    if _RE_LANG_COMPREHENSION.search(t):    return "comprehension"
    if _RE_LANG_GRAMMAR_RULE.search(t):     return "grammar_rule"
    if _RE_LANG_GRAMMAR_EXAMPLE.search(t):  return "grammar_example"
    if _RE_LANG_VOCABULARY.search(t):       return "vocabulary"
    if _RE_LANG_DIALOGUE.search(t):         return "dialogue"
    if _RE_LANG_POEM.search(t):             return "poem"
    if _RE_LANG_STORY.search(t):            return "story"
    return "body"


def _classify_sentence_length(text: str) -> str:
    """
    Classify the text block by sentence/line length characteristics.
    word_list  – mostly single words or very short phrases (vocab lists)
    short_sentence – avg sentence < 12 words (drills, gap-fill)
    long_sentence  – avg 12-25 words (explanatory sentences)
    passage        – 200+ total words (reading comprehension, story, essay)
    """
    core  = _re.sub(r"^[^\n]+\n", "", text.strip(), count=1)
    lines = [l.strip() for l in core.splitlines() if l.strip()]
    if not lines:
        return "short_sentence"
    avg   = sum(len(l.split()) for l in lines) / len(lines)
    total = sum(len(l.split()) for l in lines)
    if avg < 4 and total < 60:   return "word_list"
    if avg < 12 and total < 120: return "short_sentence"
    if total >= 200:              return "passage"
    return "long_sentence"


# Common English stop-words for keyword extraction
_STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","shall","should","may","might",
    "must","can","could","of","in","on","at","to","for","with","by","from",
    "that","this","these","those","it","its","we","our","us","you","your",
    "they","their","them","he","his","she","her","and","or","but","if","as",
    "not","no","so","also","both","each","some","any","all","more","most",
    "such","than","then","when","where","which","who","how","what","why",
    "one","two","three","into","about","after","before","between","through",
}

# ── Core classifiers ──────────────────────────────────────────────────────────

def _classify_content_type(text: str) -> str:
    """
    Return a single content_type label for a chunk based on signal patterns.
    Priority order matters — more specific signals win over general ones.
    """
    t = text.strip()

    if _RE_EXERCISE.search(t):   return "exercise"
    if _RE_SUMMARY.search(t):    return "summary"
    if _RE_ACTIVITY.search(t):   return "activity"
    if _RE_NOTE.search(t):       return "note"
    if _RE_ANSWER.search(t):     return "answer"
    if _RE_EXAMPLE.search(t):    return "example"
    if _RE_QUESTION.search(t):   return "question"
    if _RE_THINK_Q.search(t):    return "question"
    if _RE_DEFINITION.search(t): return "definition"
    if _RE_INTRO.search(t):      return "introduction"
    if _RE_TABLE.search(t):      return "table"
    if _RE_FORMULA.search(t):    return "formula"
    if _RE_FIGURE.search(t):     return "figure_ref"

    # Heuristic: short chunks (< 60 words) with a direct declarative sentence → fact
    words = t.split()
    if len(words) < 60 and _re.search(r"\b(?:is|are|was|were|has|have)\b", t):
        return "fact"

    return "body"


def _estimate_bloom_level(text: str, content_type: str) -> str:
    """
    Estimate which Bloom's taxonomy level this chunk supports answering.
    Used by the worksheet generator to fetch difficulty-appropriate content.
    """
    t = text.lower()

    # Definite higher-order signals
    if _re.search(r"\b(?:evaluate|justify|critique|design|create|hypothesi[sz]e|predict|argue|assess|compare\s+and\s+contrast|critically)\b", t):
        return "evaluate"
    if _re.search(r"\b(?:analy[sz]e|analyse|differentiate|classify|distinguish|examine|infer|investigate|why\s+does|why\s+is|what\s+would\s+happen)\b", t):
        return "analyse"
    if _re.search(r"\b(?:apply|calculate|solve|use|demonstrate|experiment|construct|show\s+that|find\s+the\s+value|derive)\b", t):
        return "apply"
    if _re.search(r"\b(?:explain|describe|summarize|interpret|discuss|illustrate|paraphrase|give\s+reason)\b", t):
        return "understand"

    # Content-type shortcuts
    if content_type in ("summary", "definition", "fact", "introduction"):
        return "remember"
    if content_type in ("exercise", "question"):
        return "apply"
    if content_type == "answer":
        return "understand"
    if content_type in ("example", "formula"):
        return "apply"

    return "remember"


def _extract_keywords(text: str, n: int = 5) -> str:
    """
    Return top-N content words as a comma-separated string.
    Used as a lightweight keyword hint in metadata.
    """
    words = _re.findall(r"\b[a-z]{4,}\b", text.lower())
    freq  = _Counter(w for w in words if w not in _STOPWORDS)
    return ", ".join(w for w, _ in freq.most_common(n))


def _parse_chapter(text: str) -> tuple[str, int]:
    """
    Find the best chapter heading in a page/chunk and return
    (chapter_label, chapter_number). Returns ("", 0) if not found.
    """
    for m in _RE_CHAPTER.finditer(text):
        # Group 1/2: "Chapter N – Title"
        if m.group(1):
            num   = int(m.group(1))
            title = (m.group(2) or "").strip()
            label = f"Chapter {num}" + (f" – {title}" if title else "")
            return label[:150], num
        # Group 3/4: "N. Title"
        if m.group(3):
            num   = int(m.group(3))
            title = (m.group(4) or "").strip()
            label = f"Chapter {num} – {title}"
            return label[:150], num
    return "", 0


def _parse_section(text: str) -> str:
    """Return the best section heading (e.g. '1.2 Types of Reactions')."""
    m = _RE_SECTION.search(text)
    if m:
        return m.group(0).strip()[:120]
    # Fallback: ALL-CAPS heading that isn't the chapter title
    m2 = _re.search(r"^([A-Z][A-Z\s]{4,60})$", text, _re.MULTILINE)
    if m2:
        return m2.group(0).strip()[:120]
    return ""


def _parse_subsection(text: str) -> str:
    """Return subsection heading (e.g. '1.2.1 Combination Reactions')."""
    m = _RE_SUBSECTION.search(text)
    return m.group(0).strip()[:120] if m else ""


# ── Main enrichment pipeline ──────────────────────────────────────────────────

def enrich_chunks(chunks: list) -> list:
    """
    Walk through all chunks in document order and attach rich metadata.

    State carried forward (sticky across chunk boundaries):
      last_chapter, last_chapter_num, last_section, last_subsection
    This ensures chunks that fall mid-chapter still know their location.
    """
    last_chapter     = ""
    last_chapter_num = 0
    last_section     = ""
    last_subsection  = ""

    for chunk in chunks:
        text = chunk.page_content

        # ── Structural location ───────────────────────────────────────────────
        ch, ch_num = _parse_chapter(text)
        sec        = _parse_section(text)
        subsec     = _parse_subsection(text)

        if ch:
            last_chapter     = ch
            last_chapter_num = ch_num
            last_section     = ""      # new chapter resets section
            last_subsection  = ""
        if sec and sec.lower() != last_chapter.lower():
            last_section    = sec
            last_subsection = ""       # new section resets subsection
        if subsec:
            last_subsection = subsec

        # ── Semantic classification ───────────────────────────────────────────
        content_type = _classify_content_type(text)
        bloom_level  = _estimate_bloom_level(text, content_type)
        keywords     = _extract_keywords(text)

        # ── Boolean signals ───────────────────────────────────────────────────
        has_formula    = bool(_RE_FORMULA.search(text))
        has_table      = bool(_RE_TABLE.search(text))
        has_figure_ref = bool(_RE_FIGURE.search(text))
        word_count     = len(text.split())

        # ── Language subject extras ───────────────────────────────────────────────
        subject        = chunk.metadata.get("subject", "")
        is_lang        = subject in _LANGUAGE_SUBJECTS
        lang_sub_type  = _classify_lang_sub_type(text) if is_lang else ""
        sentence_class = _classify_sentence_length(text) if is_lang else ""

        # ── Subject domain (science / social science sub-classification) ──────
        science_domain = _classify_science_domain(text) if subject == "science" else ""
        social_domain  = _classify_social_domain(text)  if subject == "socialscience" else ""

        # ── Question location and diagram flag ────────────────────────────────
        question_location = _classify_question_location(text)
        requires_diagram  = bool(_RE_NEEDS_DIAGRAM.search(text))

        # ── Clean chapter title (without "Chapter N –" prefix) ───────────────
        chapter_title = ""
        if last_chapter:
            m = _re.match(r"^chapter\s+\d+\s*[–\-:]\s*(.+)$", last_chapter, _re.IGNORECASE)
            chapter_title = m.group(1).strip() if m else last_chapter

        # ── Write metadata ────────────────────────────────────────────────────
        chunk.metadata.update({
            "chapter":           last_chapter,
            "chapter_num":       last_chapter_num,
            "chapter_title":     chapter_title,
            "section":           last_section,
            "subsection":        last_subsection,
            "content_type":      content_type,
            "bloom_level":       bloom_level,
            "has_formula":       has_formula,
            "has_table":         has_table,
            "has_figure_ref":    has_figure_ref,
            "requires_diagram":  requires_diagram,
            "keyword_hints":     keywords,
            "word_count":        word_count,
            # Subject domain sub-classification
            "science_domain":    science_domain,
            "social_domain":     social_domain,
            # Question location
            "question_location": question_location,
            # Language-subject fields (empty string for non-language subjects)
            "lang_sub_type":     lang_sub_type,
            "sentence_class":    sentence_class,
        })

        # ── Enrich the embedded text with structural context ──────────────────
        ctx_parts = []
        if last_chapter:       ctx_parts.append(last_chapter)
        if last_section:       ctx_parts.append(last_section)
        if last_subsection:    ctx_parts.append(last_subsection)
        ctx_parts.append(f"[{content_type}]")
        if science_domain:     ctx_parts.append(f"[{science_domain}]")
        if social_domain:      ctx_parts.append(f"[{social_domain}]")
        if question_location != "body": ctx_parts.append(f"[{question_location}]")
        if lang_sub_type:      ctx_parts.append(f"[{lang_sub_type}]")
        if sentence_class:     ctx_parts.append(f"[{sentence_class}]")

        prefix = " | ".join(ctx_parts) + "\n"
        chunk.page_content = prefix + text

    return chunks


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
        chunks = enrich_chunks(chunks)

        # ── Log metadata breakdown ────────────────────────────────────────
        from collections import Counter
        type_counts = Counter(c.metadata.get("content_type","?") for c in chunks)
        bloom_counts = Counter(c.metadata.get("bloom_level","?") for c in chunks)
        chapters_found = sorted({
            c.metadata.get("chapter","") for c in chunks if c.metadata.get("chapter")
        })

        console.print(f"   [dim]{len(pages)} pages → {len(chunks)} chunks[/dim]")

        if chapters_found:
            console.print(
                "   [green]Chapters:[/green] "
                + ", ".join(f"[cyan]{ch[:55]}[/cyan]" for ch in chapters_found[:6])
                + (f" +{len(chapters_found)-6} more" if len(chapters_found) > 6 else "")
            )

        type_str = "  ".join(
            f"[yellow]{k}[/yellow]:[white]{v}[/white]"
            for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
        )
        console.print(f"   [dim]Content types →[/dim] {type_str}")

        bloom_str = "  ".join(
            f"[magenta]{k}[/magenta]:[white]{v}[/white]"
            for k, v in sorted(bloom_counts.items(), key=lambda x: -x[1])
        )
        console.print(f"   [dim]Bloom levels  →[/dim] {bloom_str}")

        # Science domain breakdown
        if pdf_info["subject"] == "science":
            dom_counts = Counter(c.metadata.get("science_domain","?") for c in chunks)
            dom_str = "  ".join(f"[blue]{k}[/blue]:[white]{v}[/white]" for k,v in sorted(dom_counts.items(), key=lambda x: -x[1]))
            console.print(f"   [dim]Science domains →[/dim] {dom_str}")

        # Social science domain breakdown
        if pdf_info["subject"] == "socialscience":
            dom_counts = Counter(c.metadata.get("social_domain","?") for c in chunks)
            dom_str = "  ".join(f"[blue]{k}[/blue]:[white]{v}[/white]" for k,v in sorted(dom_counts.items(), key=lambda x: -x[1]))
            console.print(f"   [dim]Social domains  →[/dim] {dom_str}")

        # Question location breakdown
        loc_counts = Counter(c.metadata.get("question_location","?") for c in chunks)
        exercise_n = loc_counts.get("exercise", 0)
        intext_n   = loc_counts.get("intext", 0)
        if exercise_n or intext_n:
            console.print(f"   [dim]Questions: [green]{exercise_n} exercise[/green]  [yellow]{intext_n} in-text[/yellow][/dim]")

        # If it's a language subject, show lang_sub_type breakdown too
        if pdf_info["subject"] in {"english", "hindi", "kannada", "tamil", "sanskrit"}:
            lang_counts = Counter(c.metadata.get("lang_sub_type","?") for c in chunks if c.metadata.get("lang_sub_type"))
            if lang_counts:
                lang_str = "  ".join(
                    f"[cyan]{k}[/cyan]:[white]{v}[/white]"
                    for k, v in sorted(lang_counts.items(), key=lambda x: -x[1])
                )
                console.print(f"   [dim]Lang sub-types →[/dim] {lang_str}")

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
    chunk_size    = int(os.getenv("CHUNK_SIZE", 1000))
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