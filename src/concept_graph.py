"""
concept_graph.py
─────────────────
A structured-knowledge layer that sits ALONGSIDE the existing ChromaDB
vector store -- it never replaces retrieval, it supplements it. Every
networkx call lives in this one file. To dismantle the feature entirely:
delete this file and its two call sites (marked "CONCEPT GRAPH HOOK" in
api.py) -- nothing else in the codebase imports networkx or knows this
module exists.

Entry points used elsewhere:
  - build_from_chapter_text(...)   called once per PDF during ingestion
  - get_context_for_topics(...)    called during answer/worksheet generation
                                    (returns prose for the LLM prompt)
  - get_solvable_schema(...)       called by the question generator/solver
                                    (returns formal JSON, not prose)
  - list_computable_concepts(...)  enumerates concepts a generator can use

Storage: one JSON file per (class, subject) under CONCEPT_GRAPH_DIR, e.g.
  concept_graphs/class_10/science/graph.json
Human-readable, diffable, and trivially deletable per subject if a graph
ever needs to be rebuilt from scratch.

── Formal schema fields (added alongside the existing narrative fields) ──
For concepts that have a `formula`, the same extraction pass also asks for
a small, LLM-authored *formal* description: named variables with their
constraints, sympy-expressible relations, and difficulty levers. This is
NOT a computation engine -- it is data. Nothing in this file evaluates or
verifies math; a downstream generator/solver (sympy-based, per Nagas'
architecture) consumes get_solvable_schema()'s output to instantiate and
check problems. Concepts with no formula (history, civics, most of
language/literature) simply get empty schema fields and are invisible to
list_computable_concepts() -- this file behaves exactly as before for
non-quantitative subjects.
"""

import os
import re
import json
import time
import random
import difflib
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console

console = Console()

try:
    import networkx as nx
    from networkx.readwrite import json_graph
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

try:
    import json_repair
    _JSON_REPAIR_AVAILABLE = True
except ImportError:
    _JSON_REPAIR_AVAILABLE = False

# ── master switch ────────────────────────────────────────────────────────────
# False disables the feature everywhere it's called without touching any
# other file -- both call sites check this flag and no-op if it's False.
CONCEPT_GRAPH_ENABLED = (
    os.getenv("CONCEPT_GRAPH_ENABLED", "true").lower() == "true" and _NX_AVAILABLE
)
if os.getenv("CONCEPT_GRAPH_ENABLED", "true").lower() == "true" and not _NX_AVAILABLE:
    console.print(
        "[yellow]⚠ concept_graph: networkx not installed -- "
        "concept graph disabled. `pip install networkx` to enable.[/yellow]"
    )
if CONCEPT_GRAPH_ENABLED and not _JSON_REPAIR_AVAILABLE:
    console.print(
        "[yellow]⚠ concept_graph: json_repair not installed -- malformed LLM JSON "
        "(common on non-Latin-script extractions) will be dropped instead of repaired. "
        "`pip install json-repair` to enable (MIT licensed, pure Python, zero deps).[/yellow]"
    )

CONCEPT_GRAPH_DIR       = os.getenv("CONCEPT_GRAPH_DIR", "./concept_graphs")
MAX_GRAPH_CONTEXT_CHARS = int(os.getenv("MAX_GRAPH_CONTEXT_CHARS", 1800))
MAX_EXTRACT_CHARS       = int(os.getenv("CONCEPT_EXTRACT_CHARS", 5000))  # per LLM extraction call, Latin-script baseline
RELATION_TYPES = {"prerequisite_of", "contrasts_with", "exception_to", "applies_to"}

# ── LLM call resilience ─────────────────────────────────────────────────────
# A single chapter can spread across many windows (see below), and a
# backfill run does this back-to-back across an entire subject with no
# natural pauses -- exactly the pattern that trips a free-tier rate limit
# (429) or catches a provider backend instance mid-recycle (500). Neither
# is a data/parsing problem; both are transient and worth retrying rather
# than dropping the window. This is a fix at the LLM-call level (used by
# both api.py's live hooks and backfill_concept_graph.py), not something
# bolted onto the backfill script alone.
EXTRACT_MAX_RETRIES  = int(os.getenv("CONCEPT_EXTRACT_MAX_RETRIES", 5))
EXTRACT_RETRY_BASE_S = float(os.getenv("CONCEPT_EXTRACT_RETRY_BASE_S", 3.0))   # doubles each retry, plus jitter
EXTRACT_WINDOW_DELAY_S = float(os.getenv("CONCEPT_EXTRACT_WINDOW_DELAY_S", 1.5))  # pause before every call, not just retries
# How many windows of the SAME chapter may call the LLM at once. Previously
# windows were always processed one at a time even though nothing about a
# window depends on a prior window's result -- each is an independent LLM
# call on a different text slice. Concurrency here overlaps that dead
# waiting time instead of paying it N times sequentially. Keep modest by
# default: this multiplies with any caller-level concurrency (e.g.
# backfill_concept_graph.py running several chapters at once), so the
# effective in-flight request count is chapter_concurrency x this value --
# size both together against your provider's actual rate limit.
EXTRACT_WINDOW_CONCURRENCY = int(os.getenv("CONCEPT_EXTRACT_WINDOW_CONCURRENCY", 4))
_RETRYABLE_MARKERS = ("429", "500", "too many requests", "internal server error", "timeout", "timed out")
_corrupted_graph_backups_made = set()  # paths already backed up this process run -- avoid duplicate backup copies


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _invoke_with_retry(llm_raw, messages, context: str = ""):
    """Wraps llm_raw.invoke() with exponential backoff + jitter for
    transient provider errors (429 rate limit, 500 backend instance
    errors -- both observed in practice on NVIDIA NIM's free tier under
    sustained backfill load). Non-retryable errors (bad request, auth,
    parsing on our side) raise immediately on first attempt, same as
    before this existed."""
    last_exc = None
    for attempt in range(EXTRACT_MAX_RETRIES):
        if attempt == 0 and EXTRACT_WINDOW_DELAY_S > 0:
            time.sleep(EXTRACT_WINDOW_DELAY_S)  # preemptive pacing, not just reactive retry
        try:
            return llm_raw.invoke(messages)
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt == EXTRACT_MAX_RETRIES - 1:
                raise
            delay = EXTRACT_RETRY_BASE_S * (2 ** attempt) + random.uniform(0, 1.0)
            console.print(
                f"[yellow]concept_graph: {context} transient error "
                f"({type(e).__name__}) -- retry {attempt + 1}/{EXTRACT_MAX_RETRIES} "
                f"in {delay:.1f}s[/yellow]"
            )
            time.sleep(delay)
    raise last_exc  # pragma: no cover -- loop always returns or raises above

# ── script-aware extraction windowing ───────────────────────────────────────
# MAX_EXTRACT_CHARS above is tuned against English/Latin NCERT text. Most
# BPE tokenizers (all our providers -- DeepSeek, Groq, NIM -- included) are
# trained overwhelmingly on Latin-script corpora, so Indic scripts take
# roughly 2-4x more tokens per character. Feeding them the same char budget
# means the model's output gets silently truncated mid-JSON well before
# MAX_EXTRACT_CHARS input characters are used -- that's the root cause of
# "Expecting ',' delimiter" / "Unterminated string" failures, not a parsing
# bug. Scale the window per detected script instead of guessing once for
# Hindi; this covers any CBSE/NCERT regional-language corpus (Kannada,
# Tamil, Telugu, Bengali, etc.) without a new constant per language.
#
# Factors below are measured against the actual NIM-served tokenizer (see
# scripts/calibrate_script_budgets.py) for the scripts that were calibrated.
# Any script NOT in this table falls back to factor 1.0 (Latin baseline) in
# _extract_window_chars -- if you add NCERT content in a script not yet
# calibrated, add it to SCRIPT_RANGES and re-run the calibration script
# rather than guessing a factor.
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),   # Hindi, Marathi, Sanskrit, Nepali
    "bengali":    (0x0980, 0x09FF),   # Bengali, Assamese
    "gurmukhi":   (0x0A00, 0x0A7F),   # Punjabi
    "gujarati":   (0x0A80, 0x0AFF),
    "oriya":      (0x0B00, 0x0B7F),
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
}

SCRIPT_BUDGET_FACTOR = {
    "latin":      1.0,
    # Measured against the real nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    # tokenizer (LLM_PROVIDER=nim), via scripts/calibrate_script_budgets.py --
    # NOT estimates. Re-run that script if the NIM model ID changes, or to
    # tighten these using real NCERT paragraphs instead of single sample
    # sentences per script.
    "devanagari": 0.52,
    "bengali":    0.49,
    "gurmukhi":   0.34,
    "gujarati":   0.30,
    # oriya's measured ratio (0.37 chars/token, vs 1.6-2.7 for every other
    # Indic script here) is anomalous -- likely the tokenizer falling back to
    # byte-level encoding for weak/no native Oriya vocab coverage, not a
    # genuinely denser script. Floored at 0.15 as a safety margin. If you
    # ingest real Odia NCERT chapters, spot-check tok.encode() output on an
    # actual paragraph before trusting this number in production.
    "oriya":      0.15,
    "tamil":      0.46,
    "telugu":     0.40,
    "kannada":    0.45,
    "malayalam":  0.39,
}


def _detect_dominant_script(text: str, sample_size: int = 3000) -> str:
    """Cheap, dependency-free (stdlib codepoint ranges) script detector.
    Samples the first `sample_size` chars -- chapter text is fairly
    homogeneous in script, so a prefix sample is enough to pick a window
    size and avoids scanning very long chapters char-by-char."""
    sample = text[:sample_size]
    counts = {name: 0 for name in SCRIPT_RANGES}
    total_scripted = 0
    for ch in sample:
        cp = ord(ch)
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[name] += 1
                total_scripted += 1
                break
    if total_scripted == 0:
        return "latin"
    dominant = max(counts, key=counts.get)
    # require the dominant non-Latin script to be a meaningful share of the
    # sample, not just a stray character or two, before trusting it
    if counts[dominant] / max(len(sample), 1) < 0.05:
        return "latin"
    return dominant


def _extract_window_chars(text: str) -> int:
    script = _detect_dominant_script(text)
    factor = SCRIPT_BUDGET_FACTOR.get(script, 1.0)
    return max(int(MAX_EXTRACT_CHARS * factor), 500)  # floor so windows can't collapse to ~nothing

EXTRACTION_PROMPT = """You are building a structured concept map for a CBSE
textbook chapter. Read the CHAPTER TEXT below and extract the distinct
concepts it teaches.

For EACH concept, return:
- "name": short concept name (e.g. "Ohm's Law")
- "definition": one precise sentence
- "key_facts": 2-4 specific facts a question could test
- "formula": the formula/rule if this concept has one, else null
- "common_misconceptions": 1-3 things students commonly get WRONG about this
  concept specifically -- must be plausible, specific errors, not generic
  ("students may forget the formula" is USELESS; "students often confuse
  resistance increasing with temperature in conductors vs. decreasing in
  semiconductors" is USEFUL)
- "applications": 1-2 real-world or exam-style application scenarios
- "relations": list of {{"target": "<other concept name in this chapter>",
  "type": "<one of prerequisite_of|contrasts_with|exception_to|applies_to>"}}
  -- only include relations you are confident about; target must be another
  concept name, either one you're also extracting here or a well-known
  concept from an earlier part of this same subject.

IF AND ONLY IF this concept has a "formula" (i.e. formula is not null),
ALSO fill in these four fields so the formula can be used to generate and
verify problems, not just explain the concept. If formula is null, set all
four to empty ({{}} or []) -- do not invent a formal schema for a concept
that has no formula.
- "variables": object mapping each symbol in the formula to a short plain-
  English constraint, e.g. {{"a": "nonzero real", "n": "positive integer"}}
- "relations_formal": list of the formula's relations written as plain
  algebraic strings a symbolic math library could parse, e.g.
  ["sum_roots = -b/a", "product_roots = c/a"]. Use the same variable names
  as in "variables". Only include relations that follow directly and
  unambiguously from the formula itself -- do not derive new identities.
- "constraint_templates": list of validity conditions as plain strings,
  e.g. ["a != 0", "discriminant >= 0 for real roots"]
- "difficulty_levers": 1-4 short phrases describing ways a question on this
  concept can be made harder or reframed, e.g. ["give roots, ask for the
  equation (reverse direction)", "irrational or negative roots",
  "combine with the discriminant condition"] -- describe *what varies*,
  not a worked example.

SEPARATELY, IF AND ONLY IF this concept is about classifying/grouping
things into distinct types or categories based on shared properties (e.g.
"Acids, Bases and Salts", "Types of Soil", "Living and Non-living Things",
"Classification of Plants") -- NOT concepts that merely mention an example
of something in passing -- ALSO fill in these four fields. A concept can
have EITHER the four formula fields above OR these four, rarely both, and
usually neither. If not a classification concept, set all four to empty
({{}} or "").
- "categories": list of the distinct category names, e.g. ["acids",
  "bases", "salts"]
- "classification_criteria": one plain sentence stating the rule used to
  decide which category something belongs to, e.g. "based on whether the
  substance releases H+ or OH- ions in water"
- "category_features": object mapping each category name to 2-4 short
  defining features/properties, e.g. {{"acids": ["turns blue litmus red",
  "pH < 7"], "bases": ["turns red litmus blue", "pH > 7"]}}
- "known_examples": object mapping each category name to 2-5 specific
  example items drawn from the chapter text, e.g. {{"acids": ["lemon
  juice", "vinegar"], "bases": ["baking soda", "soap"]}}

Return ONLY valid JSON -- a single JSON array, no markdown fences, no preamble,
no trailing commentary. Every backslash inside a string must be a valid JSON
escape (\\\\, \\", \\n, \\t, \\uXXXX) -- never emit LaTeX-style escapes like
\\frac or \\times, and never escape a non-ASCII letter; write formulas and
non-English text as plain literal characters instead.
[
  {{
    "name": "...",
    "definition": "...",
    "key_facts": ["...", "..."],
    "formula": null,
    "common_misconceptions": ["..."],
    "applications": ["..."],
    "relations": [{{"target": "...", "type": "contrasts_with"}}],
    "variables": {{"a": "nonzero real"}},
    "relations_formal": ["sum_roots = -b/a"],
    "constraint_templates": ["a != 0"],
    "difficulty_levers": ["reverse: give roots, ask for the equation"],
    "categories": [],
    "classification_criteria": "",
    "category_features": {{}},
    "known_examples": {{}}
  }},
  {{
    "name": "Acids and Bases",
    "definition": "...",
    "key_facts": ["...", "..."],
    "formula": null,
    "common_misconceptions": ["..."],
    "applications": ["..."],
    "relations": [],
    "variables": {{}},
    "relations_formal": [],
    "constraint_templates": [],
    "difficulty_levers": ["borderline pH near 7", "everyday item classified by an unstated property"],
    "categories": ["acids", "bases"],
    "classification_criteria": "based on whether the substance releases H+ or OH- ions in water",
    "category_features": {{"acids": ["turns blue litmus red", "pH < 7"], "bases": ["turns red litmus blue", "pH > 7"]}},
    "known_examples": {{"acids": ["lemon juice", "vinegar"], "bases": ["baking soda", "soap"]}}
  }}
]

Extract at most 12 concepts. Skip trivial/administrative text (page
headers, "Figure it out" labels with no content, etc).

CHAPTER TEXT:
{text}"""


# ── id / slug helpers ─────────────────────────────────────────────────────────
def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return s.strip("_")[:60] or "unnamed"


def _node_id(subject: str, chapter: str, name: str) -> str:
    return f"{_slug(subject)}::{_slug(chapter)}::{_slug(name)}"


def _graph_path(class_name: str, subject: str) -> Path:
    return Path(CONCEPT_GRAPH_DIR) / class_name / subject / "graph.json"


# ── load / save ────────────────────────────────────────────────────────────────
def _load_graph(class_name: str, subject: str):
    if not CONCEPT_GRAPH_ENABLED:
        return None
    path = _graph_path(class_name, subject)
    if not path.exists():
        return nx.DiGraph()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return json_graph.node_link_graph(data, edges="edges")
    except TypeError:
        # older networkx versions use `link` instead of `edges` kwarg
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)
    except Exception as e:
        # NEVER silently fall back to an empty graph here. This graph gets
        # merged into and saved right back over `path` -- returning an
        # empty graph on a parse failure means the next _save_graph() call
        # permanently overwrites whatever was recoverable in the corrupted
        # file with a much smaller "fresh" one. That's silent, irreversible
        # data loss, and it already happened once (see the incident this
        # guard is a response to).
        #
        # Copy (not move) the corrupted file to a timestamped backup and
        # leave the original in place at `path`. This is deliberate: if we
        # moved it away, the NEXT chapter in this subject would see
        # path.exists() == False and silently start fresh too, so only the
        # first chapter to hit the corruption would ever surface an error
        # -- every other chapter in the same run would quietly repeat the
        # exact data loss this guard exists to prevent. Leaving the
        # corrupted file in place means every chapter in this subject keeps
        # failing loudly and consistently until a human actually fixes it.
        backup = path.with_name(f"{path.stem}.corrupted.{int(time.time())}{path.suffix}")
        if path in _corrupted_graph_backups_made:
            console.print(
                f"[bold red]concept_graph: {path} is still corrupted ({e}) -- "
                f"already backed up earlier this run, refusing to continue "
                f"for this subject.[/bold red]"
            )
        else:
            try:
                import shutil
                shutil.copy2(path, backup)
                _corrupted_graph_backups_made.add(path)
                console.print(
                    f"[bold red]concept_graph: {path} is corrupted ({e}) -- "
                    f"backed up to {backup} for manual recovery. NOT starting "
                    f"fresh, since that would silently discard this subject's "
                    f"existing graph on the next save. Refusing to continue "
                    f"for this subject until it's fixed -- repair or restore "
                    f"{path}, then re-run.[/bold red]"
                )
            except OSError as backup_err:
                console.print(
                    f"[bold red]concept_graph: {path} is corrupted ({e}) AND "
                    f"could not be backed up ({backup_err}) -- refusing to "
                    f"proceed to avoid silent data loss.[/bold red]"
                )
        raise RuntimeError(
            f"concept_graph: refusing to continue for {class_name}/{subject} -- "
            f"{path} failed to parse and has been backed up rather than discarded. "
            f"See console output above for the backup path."
        ) from e


def _save_graph(graph, class_name: str, subject: str):
    path = _graph_path(class_name, subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json_graph.node_link_data(graph, edges="edges")
    except TypeError:
        data = json_graph.node_link_data(graph)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic-ish write, avoids truncated files on crash


# ── ingestion-time build ────────────────────────────────────────────────────────
def _extract_one_window(window: str, wi: int, chapter: str, llm_raw):
    """Runs one window's LLM extraction call + JSON parse/repair. No graph
    access here at all -- this is the part safe to run concurrently across
    windows AND across chapters, since it touches nothing shared."""
    raw = ""
    try:
        from langchain_core.messages import HumanMessage
        prompt = EXTRACTION_PROMPT.format(text=window)
        response = _invoke_with_retry(
            llm_raw, [HumanMessage(content=prompt)],
            context=f"'{chapter}' window {wi}",
        )
        raw = (response.content or "").strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return wi, json.loads(raw), False
    except Exception as e:
        if _JSON_REPAIR_AVAILABLE and raw:
            try:
                concepts = json_repair.loads(raw)
                console.print(
                    f"[yellow]concept_graph: window {wi} needed repair "
                    f"({type(e).__name__}: {e}) -- salvaged via json_repair[/yellow]"
                )
                return wi, concepts, False
            except Exception as e2:
                console.print(
                    f"[red]concept_graph: window {wi} failed even after repair: {e2}[/red]"
                )
                return wi, None, True
        console.print(f"[red]concept_graph: extraction window {wi} failed: {e}[/red]")
        return wi, None, True


def _compute_windows(chapter_text: str) -> tuple:
    """Deterministic split of chapter_text into script-aware windows.
    Given the same chapter_text, always returns the same windows in the
    same order -- this is what makes targeted re-extraction of specific
    failed window indices possible without re-running the whole chapter."""
    window_chars = _extract_window_chars(chapter_text)
    script = _detect_dominant_script(chapter_text)
    windows = [
        chapter_text[i:i + window_chars]
        for i in range(0, len(chapter_text), window_chars)
    ] or [chapter_text]
    return windows, script, window_chars


def extract_concepts_from_chapter(chapter_text: str, chapter: str, llm_raw) -> dict:
    """
    Phase 1: pure LLM extraction, no graph.json read/write at all -- safe to
    run for many chapters (and many windows within a chapter) at the same
    time with no locking, since nothing here touches shared state. Returns
    the raw concept list plus bookkeeping, including which specific window
    indices failed (not just a count) -- pass those to
    retry_failed_windows(...) to re-extract only what's missing instead of
    redoing the whole chapter. Call merge_and_save_chapter(...) next to
    actually persist the result.
    """
    if not chapter_text or not chapter_text.strip():
        return {"status": "skipped", "reason": "empty chapter text", "concepts": []}

    console.print(f"[cyan]concept_graph: extracting concepts from '{chapter}'…[/cyan]")

    windows, script, window_chars = _compute_windows(chapter_text)
    if script != "latin":
        console.print(
            f"[cyan]concept_graph: detected '{script}' script -- using "
            f"{window_chars}-char windows (vs {MAX_EXTRACT_CHARS} baseline)[/cyan]"
        )

    results = [None] * len(windows)
    failed_indices = []
    max_workers = max(1, min(EXTRACT_WINDOW_CONCURRENCY, len(windows)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_extract_one_window, window, wi, chapter, llm_raw): wi
            for wi, window in enumerate(windows)
        }
        for future in as_completed(futures):
            wi, concepts, dropped = future.result()
            if dropped:
                failed_indices.append(wi)
            else:
                results[wi] = concepts

    all_concepts = []
    for concepts in results:
        if isinstance(concepts, list):
            all_concepts.extend(concepts)

    if failed_indices:
        console.print(
            f"[yellow]concept_graph: '{chapter}' -- {len(failed_indices)}/{len(windows)} "
            f"windows dropped, concept graph for this chapter is incomplete[/yellow]"
        )

    return {
        "status": "ok" if not failed_indices else "partial",
        "concepts": all_concepts,
        "windows_total": len(windows),
        "windows_dropped": len(failed_indices),
        "failed_window_indices": sorted(failed_indices),
        "detected_script": script,
    }


def retry_failed_windows(chapter_text: str, chapter: str, llm_raw, failed_window_indices: list) -> dict:
    """
    Re-extracts ONLY the specific windows that failed last time, instead of
    the whole chapter. Recomputes the same deterministic window split (see
    _compute_windows) and re-runs just the requested indices. Returns the
    same shape as extract_concepts_from_chapter -- "concepts" here is only
    the NEWLY recovered concepts, meant to be merged on top of what was
    already saved from the first pass (merge_and_save_chapter dedups, so
    this is safe to merge in without re-adding the successful windows).
    """
    windows, script, _ = _compute_windows(chapter_text)
    targets = [wi for wi in failed_window_indices if 0 <= wi < len(windows)]
    if not targets:
        return {"status": "ok", "concepts": [], "windows_total": len(windows),
                "windows_dropped": 0, "failed_window_indices": [], "detected_script": script}

    results = {}
    still_failed = []
    max_workers = max(1, min(EXTRACT_WINDOW_CONCURRENCY, len(targets)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_extract_one_window, windows[wi], wi, chapter, llm_raw): wi
            for wi in targets
        }
        for future in as_completed(futures):
            wi, concepts, dropped = future.result()
            if dropped:
                still_failed.append(wi)
            else:
                results[wi] = concepts

    all_concepts = []
    for wi in targets:
        if wi in results and isinstance(results[wi], list):
            all_concepts.extend(results[wi])

    return {
        "status": "ok" if not still_failed else "partial",
        "concepts": all_concepts,
        "windows_total": len(windows),
        "windows_dropped": len(still_failed),
        "failed_window_indices": sorted(still_failed),
        "detected_script": script,
    }


def merge_and_save_chapter(
    extracted: dict,
    class_name: str,
    subject:    str,
    chapter:    str,
    source:     str = "",
) -> dict:
    """
    Phase 2: the only part that touches graph.json. Callers running many
    chapters concurrently (see backfill_concept_graph.py) MUST serialize
    calls to this function per (class_name, subject) -- it does a
    load -> merge -> save read-modify-write on one shared file, and two
    concurrent callers writing the same file will race and silently drop
    one call's concepts. Keep this function's body fast (no LLM calls, no
    network I/O) so that lock is held as briefly as possible.
    """
    if extracted.get("status") in ("disabled", "skipped"):
        return extracted

    graph = _load_graph(class_name, subject)
    n_c, n_r = _merge_concepts(graph, extracted.get("concepts") or [], subject, chapter, source)
    _save_graph(graph, class_name, subject)

    console.print(
        f"[green]concept_graph: '{chapter}' → {n_c} concepts, "
        f"{n_r} relations (graph now {graph.number_of_nodes()} "
        f"nodes total for {class_name}/{subject})[/green]"
    )
    return {
        "status": extracted.get("status", "ok"),
        "concepts_added_or_updated": n_c,
        "relations_added": n_r,
        "graph_total_nodes": graph.number_of_nodes(),
        "windows_total": extracted.get("windows_total", 0),
        "windows_dropped": extracted.get("windows_dropped", 0),
        "failed_window_indices": extracted.get("failed_window_indices", []),
        "detected_script": extracted.get("detected_script", "latin"),
    }


def build_from_chapter_text(
    chapter_text: str,
    class_name:   str,
    subject:      str,
    chapter:      str,
    llm_raw,                       # unbound LLM client -- caller supplies it
    source:       str = "",
) -> dict:
    """
    Called once per ingested PDF (api.py's live single-file upload path).
    Extracts concepts + relations via one or more LLM calls (windowed and
    run concurrently if the chapter is long) and merges them into the
    persistent (class, subject) graph. Never raises -- returns a dict with
    an "error"/"status" key on failure so ingestion itself is never blocked
    by a concept-graph problem.

    This is just extract_concepts_from_chapter() + merge_and_save_chapter()
    run back to back -- callers doing many chapters concurrently (backfill)
    should call those two functions directly instead, so only the merge
    step needs a lock instead of the whole thing.
    """
    if not CONCEPT_GRAPH_ENABLED:
        return {"status": "disabled"}

    extracted = extract_concepts_from_chapter(chapter_text, chapter, llm_raw)
    if extracted.get("status") == "skipped":
        return extracted
    return merge_and_save_chapter(extracted, class_name, subject, chapter, source)


def _merge_concepts(graph, concepts, subject: str, chapter: str, source: str) -> tuple:
    """Add/update nodes and edges in-place. Returns (n_concepts, n_relations)."""
    if not isinstance(concepts, list):
        return 0, 0

    now = datetime.now(timezone.utc).isoformat()
    name_to_id = {}  # local name -> node id, for resolving relation targets in this batch

    n_concepts = 0
    for c in concepts:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        name = c["name"].strip()
        nid  = _node_id(subject, chapter, name)
        name_to_id[name.lower()] = nid

        existing = graph.nodes.get(nid, {})
        graph.add_node(
            nid,
            name=name,
            chapter=chapter,
            subject=subject,
            definition=c.get("definition") or existing.get("definition", ""),
            key_facts=_dedup_extend(existing.get("key_facts", []), c.get("key_facts") or []),
            formula=c.get("formula") or existing.get("formula"),
            common_misconceptions=_dedup_extend(
                existing.get("common_misconceptions", []), c.get("common_misconceptions") or []
            ),
            applications=_dedup_extend(existing.get("applications", []), c.get("applications") or []),
            # Formal schema fields -- only ever populated when the concept
            # has a formula (enforced by the extraction prompt, not here).
            # variables is a dict merge (new keys win on conflict, since a
            # later/more-detailed extraction pass is more likely correct);
            # the three list fields dedupe the same way as key_facts etc.
            variables=_merge_variable_dict(existing.get("variables", {}), c.get("variables") or {}),
            relations_formal=_dedup_extend(
                existing.get("relations_formal", []), c.get("relations_formal") or []
            ),
            constraint_templates=_dedup_extend(
                existing.get("constraint_templates", []), c.get("constraint_templates") or []
            ),
            difficulty_levers=_dedup_extend(
                existing.get("difficulty_levers", []), c.get("difficulty_levers") or []
            ),
            # Classification schema fields -- only ever populated when the
            # concept is a classification/grouping concept (enforced by the
            # extraction prompt, not here). classification_criteria is a
            # single sentence so it overwrites like 'definition'; categories
            # dedupes like a normal list; category_features/known_examples
            # are dict-of-list, so each category's list is dedup-merged
            # independently (a later pass may add new examples for an
            # already-known category without dropping the earlier ones).
            classification_criteria=(
                c.get("classification_criteria") or existing.get("classification_criteria", "")
            ),
            categories=_dedup_extend(existing.get("categories", []), c.get("categories") or []),
            category_features=_merge_dict_of_lists(
                existing.get("category_features", {}), c.get("category_features") or {}
            ),
            known_examples=_merge_dict_of_lists(
                existing.get("known_examples", {}), c.get("known_examples") or {}
            ),
            source=source or existing.get("source", ""),
            stub=False,
            updated_at=now,
        )
        n_concepts += 1

    n_relations = 0
    for c in concepts:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        src_id = name_to_id.get(c["name"].strip().lower())
        for rel in (c.get("relations") or []):
            if not isinstance(rel, dict):
                continue
            target_name = (rel.get("target") or "").strip()
            rel_type    = rel.get("type")
            if not target_name or rel_type not in RELATION_TYPES:
                continue
            tgt_id = name_to_id.get(target_name.lower()) or _node_id(subject, chapter, target_name)
            if tgt_id not in graph:
                # Referenced but not (yet) extracted in detail -- add a stub
                # so the edge is meaningful now and gets filled in if/when
                # that concept is extracted from a later chapter/PDF.
                graph.add_node(tgt_id, name=target_name, chapter=chapter, subject=subject,
                                definition="", key_facts=[], formula=None,
                                common_misconceptions=[], applications=[],
                                source="", stub=True, updated_at=now)
            if not graph.has_edge(src_id, tgt_id):
                graph.add_edge(src_id, tgt_id, type=rel_type)
                n_relations += 1

    return n_concepts, n_relations


def _merge_variable_dict(existing: dict, new_vars: dict) -> dict:
    """Merge the 'variables' schema field. Not a list, so _dedup_extend
    doesn't apply -- new descriptions win on key conflict (a later
    extraction pass re-reading the same formula is treated as more likely
    correct, same rationale as the 'formula' field's overwrite-on-reextract
    behavior above)."""
    if not isinstance(existing, dict):
        existing = {}
    merged = dict(existing)
    if isinstance(new_vars, dict):
        for k, v in new_vars.items():
            if k and v:
                merged[str(k)] = v
    return merged


def _merge_dict_of_lists(existing: dict, new_vals: dict) -> dict:
    """Merge dict-of-list schema fields (category_features, known_examples).
    Each key (category name) is merged independently with _dedup_extend, so
    re-extracting the same chapter adds new examples/features under a
    category without duplicating or dropping earlier ones."""
    if not isinstance(existing, dict):
        existing = {}
    merged = {k: list(v) for k, v in existing.items()}
    if isinstance(new_vals, dict):
        for k, v in new_vals.items():
            if not k:
                continue
            merged[str(k)] = _dedup_extend(merged.get(str(k), []), v or [])
    return merged


def _dedup_extend(existing: list, new_items: list) -> list:
    seen = {str(x).strip().lower() for x in existing}
    out = list(existing)
    for item in new_items:
        key = str(item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ── generation-time retrieval ───────────────────────────────────────────────────
def _match_nodes(graph, topic: str, top_k: int = 2) -> list:
    """Fuzzy-match a topic string against node names. Cheap, dependency-free
    (difflib is stdlib) -- swap for embedding similarity later if match
    quality against exam-phrased/OCR-noisy topics turns out to need it."""
    topic_norm = topic.strip().lower()
    if not topic_norm:
        return []

    names = {nid: (data.get("name") or "").lower() for nid, data in graph.nodes(data=True)}

    # exact / substring hits first
    substring_hits = [nid for nid, name in names.items() if name and (name in topic_norm or topic_norm in name)]
    if substring_hits:
        return substring_hits[:top_k]

    # fall back to fuzzy ratio match
    close = difflib.get_close_matches(topic_norm, list(names.values()), n=top_k, cutoff=0.55)
    return [nid for nid, name in names.items() if name in close][:top_k]


def _format_node_block(graph, node_id: str, include_neighbors: bool = True) -> str:
    data = graph.nodes[node_id]
    if data.get("stub"):
        return ""  # nothing useful to say about a stub yet

    lines = [f"Concept: {data.get('name')} (chapter: {data.get('chapter')})"]
    if data.get("definition"):
        lines.append(f"Definition: {data['definition']}")
    if data.get("formula"):
        lines.append(f"Formula: {data['formula']}")
    if data.get("key_facts"):
        lines.append("Key facts: " + "; ".join(data["key_facts"][:4]))
    if data.get("common_misconceptions"):
        lines.append("Common student misconceptions: " + "; ".join(data["common_misconceptions"][:3]))
    if data.get("applications"):
        lines.append("Applications: " + "; ".join(data["applications"][:2]))

    if include_neighbors:
        for _, tgt, edata in graph.out_edges(node_id, data=True):
            tname = graph.nodes[tgt].get("name")
            if tname and not graph.nodes[tgt].get("stub"):
                lines.append(f"  [{edata.get('type')}] → {tname}")
        for src, _, edata in graph.in_edges(node_id, data=True):
            sname = graph.nodes[src].get("name")
            if sname and not graph.nodes[src].get("stub"):
                lines.append(f"  ← [{edata.get('type')}] {sname}")

    return "\n".join(lines)


def get_context_for_topics(
    topics:            list,
    class_name:         str,
    subject:            str,
    top_k_per_topic:    int = 2,
) -> str:
    """
    Called from api.py's answer_stream / worksheet_stream. Returns a plain
    text block to append alongside (never instead of) the Chroma-retrieved
    context, or "" if the graph is disabled / empty / nothing matched --
    callers should treat "" as "no-op, proceed exactly as before".
    """
    if not CONCEPT_GRAPH_ENABLED or not topics:
        return ""

    graph = _load_graph(class_name, subject)
    if graph is None or graph.number_of_nodes() == 0:
        return ""

    blocks, matched = [], set()
    for topic in topics:
        for node_id in _match_nodes(graph, topic, top_k_per_topic):
            if node_id in matched:
                continue
            matched.add(node_id)
            block = _format_node_block(graph, node_id)
            if block:
                blocks.append(block)

    if not blocks:
        return ""

    text = "\n\n".join(blocks)
    return text[:MAX_GRAPH_CONTEXT_CHARS]



# ── formal schema types ─────────────────────────────────────────────────────
# Adding a new schema family (e.g. causal_timeline, process_sequence) later
# means: (1) a new instruction block in EXTRACTION_PROMPT, (2) new fields
# merged in _merge_concepts, (3) one entry here. get_schema/list_schema_
# concepts below don't need to change -- this is the one place that knows
# how many schema types exist.
_SCHEMA_TYPE_FIELDS = {
    # schema_type: [(field_name, default_factory_or_None), ...]
    # The FIRST field in each list is the "key field": a concept only
    # counts as usable for this type if that field is non-empty.
    "computable": [
        ("relations_formal",     list),
        ("formula",              None),
        ("variables",            dict),
        ("constraint_templates", list),
        ("difficulty_levers",    list),
    ],
    "classification": [
        ("categories",              list),
        ("classification_criteria", str),
        ("category_features",       dict),
        ("known_examples",          dict),
        ("difficulty_levers",       list),
    ],
}


def _schema_from_node(data: dict, schema_type: str) -> dict:
    out = {"concept": data.get("name"), "chapter": data.get("chapter"), "schema_type": schema_type}
    for field, default in _SCHEMA_TYPE_FIELDS[schema_type]:
        out[field] = data.get(field) if default is None else data.get(field, default())
    return out


def get_schema(topic: str, class_name: str, subject: str, schema_type: str = "computable", top_k: int = 1) -> list:
    """
    For a future generator/solver, NOT for the answer/worksheet prompt --
    returns formal JSON, not prose. Callers pass this straight to the
    schema_type-appropriate downstream code (sympy for "computable", a
    category-match checker for "classification", etc.); nothing in this
    file interprets or evaluates the strings inside.

    Returns a list of schema dicts (usually 0 or 1 items; top_k>1 only
    matters if a topic string fuzzy-matches multiple concepts). Skips any
    concept missing that schema_type's key field -- e.g. a History concept
    correctly returns [] for schema_type="computable", and a Physics
    formula concept correctly returns [] for schema_type="classification".
    """
    if schema_type not in _SCHEMA_TYPE_FIELDS:
        raise ValueError(f"Unknown schema_type {schema_type!r}. Known: {list(_SCHEMA_TYPE_FIELDS)}")
    if not CONCEPT_GRAPH_ENABLED or not topic:
        return []

    graph = _load_graph(class_name, subject)
    if graph is None or graph.number_of_nodes() == 0:
        return []

    key_field = _SCHEMA_TYPE_FIELDS[schema_type][0][0]
    out = []
    for node_id in _match_nodes(graph, topic, top_k):
        data = graph.nodes[node_id]
        if data.get("stub") or not data.get(key_field):
            continue
        out.append(_schema_from_node(data, schema_type))
    return out


def list_schema_concepts(class_name: str, subject: str, schema_type: str = "computable") -> list:
    """
    Enumerates every concept in this (class, subject) graph usable for the
    given schema_type -- i.e. what a question generator could pick from
    without the caller needing to guess topic strings up front. Same shape
    as get_schema()'s list items.
    """
    if schema_type not in _SCHEMA_TYPE_FIELDS:
        raise ValueError(f"Unknown schema_type {schema_type!r}. Known: {list(_SCHEMA_TYPE_FIELDS)}")
    if not CONCEPT_GRAPH_ENABLED:
        return []

    graph = _load_graph(class_name, subject)
    if graph is None:
        return []

    key_field = _SCHEMA_TYPE_FIELDS[schema_type][0][0]
    out = []
    for node_id, data in graph.nodes(data=True):
        if data.get("stub") or not data.get(key_field):
            continue
        out.append(_schema_from_node(data, schema_type))
    return out


# Back-compat aliases -- everything above generalized what used to be two
# computable-only functions. Nothing downstream has been built against
# these yet, but keeping the names stable costs nothing and avoids a
# needless rename churn if that changes.
def get_solvable_schema(topic: str, class_name: str, subject: str, top_k: int = 1) -> list:
    return get_schema(topic, class_name, subject, schema_type="computable", top_k=top_k)


def list_computable_concepts(class_name: str, subject: str) -> list:
    return list_schema_concepts(class_name, subject, schema_type="computable")


def get_classification_schema(topic: str, class_name: str, subject: str, top_k: int = 1) -> list:
    return get_schema(topic, class_name, subject, schema_type="classification", top_k=top_k)


def list_classification_concepts(class_name: str, subject: str) -> list:
    return list_schema_concepts(class_name, subject, schema_type="classification")


def match_count(topics: list, class_name: str, subject: str) -> int:
    """Lightweight helper for the SSE 'graph' status event -- lets the UI
    show whether the graph actually contributed anything, without callers
    needing to know graph internals."""
    if not CONCEPT_GRAPH_ENABLED or not topics:
        return 0
    graph = _load_graph(class_name, subject)
    if graph is None:
        return 0
    matched = set()
    for topic in topics:
        matched.update(_match_nodes(graph, topic, 2))
    return len([n for n in matched if not graph.nodes[n].get("stub")])
