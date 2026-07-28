"""
concept_graph.py
─────────────────
A structured-knowledge layer that sits ALONGSIDE the existing ChromaDB
vector store -- it never replaces retrieval, it supplements it. Every
networkx call lives in this one file. To dismantle the feature entirely:
delete this file and its two call sites (marked "CONCEPT GRAPH HOOK" in
api.py) -- nothing else in the codebase imports networkx or knows this
module exists.

Two entry points used elsewhere:
  - build_from_chapter_text(...)   called once per PDF during ingestion
  - get_context_for_topics(...)    called during answer/worksheet generation

Storage: one JSON file per (class, subject) under CONCEPT_GRAPH_DIR, e.g.
  concept_graphs/class_10/science/graph.json
Human-readable, diffable, and trivially deletable per subject if a graph
ever needs to be rebuilt from scratch.
"""

import os
import re
import json
import difflib
from pathlib import Path
from datetime import datetime, timezone

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
    "relations": [{{"target": "...", "type": "contrasts_with"}}]
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
        console.print(f"[red]concept_graph: failed to load {path}: {e} -- starting fresh[/red]")
        return nx.DiGraph()


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
def build_from_chapter_text(
    chapter_text: str,
    class_name:   str,
    subject:      str,
    chapter:      str,
    llm_raw,                       # unbound LLM client -- caller supplies it
    source:       str = "",
) -> dict:
    """
    Called once per ingested PDF. Extracts concepts + relations via one or
    more LLM calls (windowed if the chapter is long) and merges them into
    the persistent (class, subject) graph. Never raises -- returns a dict
    with an "error" key on failure so ingestion itself is never blocked by
    a concept-graph problem.
    """
    if not CONCEPT_GRAPH_ENABLED:
        return {"status": "disabled"}
    if not chapter_text or not chapter_text.strip():
        return {"status": "skipped", "reason": "empty chapter text"}

    console.print(f"[cyan]concept_graph: extracting concepts from '{chapter}' ({subject})…[/cyan]")

    graph = _load_graph(class_name, subject)
    total_concepts = 0
    total_relations = 0

    # Window long chapters so each extraction call stays well within
    # context and the model doesn't silently truncate its own output.
    # Window size is script-aware (see SCRIPT_BUDGET_FACTOR above) rather
    # than a flat char count, since non-Latin scripts overrun a
    # Latin-tuned budget in tokens well before the char limit is hit.
    window_chars = _extract_window_chars(chapter_text)
    script = _detect_dominant_script(chapter_text)
    if script != "latin":
        console.print(
            f"[cyan]concept_graph: detected '{script}' script -- using "
            f"{window_chars}-char windows (vs {MAX_EXTRACT_CHARS} baseline)[/cyan]"
        )
    windows = [
        chapter_text[i:i + window_chars]
        for i in range(0, len(chapter_text), window_chars)
    ] or [chapter_text]

    dropped_windows = 0
    for wi, window in enumerate(windows):
        raw = ""
        try:
            from langchain_core.messages import HumanMessage
            prompt = EXTRACTION_PROMPT.format(text=window)
            response = llm_raw.invoke([HumanMessage(content=prompt)])
            raw = (response.content or "").strip()
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                raw = match.group(0)
            concepts = json.loads(raw)
        except Exception as e:
            # Strict parse failed -- try a repair pass before giving up on
            # this window's concepts entirely. json_repair handles both
            # truncated JSON (unterminated strings/arrays) and invalid
            # escapes; it's a best-effort salvage, not a guarantee.
            if _JSON_REPAIR_AVAILABLE and raw:
                try:
                    concepts = json_repair.loads(raw)
                    console.print(
                        f"[yellow]concept_graph: window {wi} needed repair "
                        f"({type(e).__name__}: {e}) -- salvaged via json_repair[/yellow]"
                    )
                except Exception as e2:
                    console.print(
                        f"[red]concept_graph: window {wi} failed even after repair: {e2}[/red]"
                    )
                    dropped_windows += 1
                    continue
            else:
                console.print(f"[red]concept_graph: extraction window {wi} failed: {e}[/red]")
                dropped_windows += 1
                continue

        n_c, n_r = _merge_concepts(graph, concepts, subject, chapter, source)
        total_concepts += n_c
        total_relations += n_r

    if dropped_windows:
        console.print(
            f"[yellow]concept_graph: '{chapter}' -- {dropped_windows}/{len(windows)} "
            f"windows dropped, concept graph for this chapter is incomplete[/yellow]"
        )

    _save_graph(graph, class_name, subject)
    console.print(
        f"[green]concept_graph: '{chapter}' → {total_concepts} concepts, "
        f"{total_relations} relations (graph now {graph.number_of_nodes()} "
        f"nodes total for {class_name}/{subject})[/green]"
    )
    return {
        "status": "ok" if dropped_windows == 0 else "partial",
        "concepts_added_or_updated": total_concepts,
        "relations_added": total_relations,
        "graph_total_nodes": graph.number_of_nodes(),
        "windows_total": len(windows),
        "windows_dropped": dropped_windows,
        "detected_script": script,
    }


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
