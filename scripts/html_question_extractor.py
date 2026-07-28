"""
Extracts questions from HTML files, including HTML exports from OCR tools
(e.g. Mathpix) where math notation is rendered as MathJax SVG glyphs.

Why this works without OCR/vision:
Mathpix-style exports render each formula 3 ways in parallel, all as siblings
inside a <span class="math-inline"> or <span class="math-block">:
  - <mathml>      (hidden, MathML)
  - <mathmlword>  (hidden, MathML w/ real unicode)
  - <asciimath>   (hidden, AsciiMath)
  - <latex>       (hidden, LaTeX source)
  - <mjx-container><svg>...</svg></mjx-container>  (visible glyph outlines)

The visible SVG is glyph paths only — not machine-readable and not worth
parsing. We instead pull the <latex> sibling, which is exact and clean, and
substitute it inline as $...$ (inline math) or $$...$$ (block/display math).
This keeps LLM-facing question text in a form models are heavily trained on.

If a math-inline/math-block span has no <latex> sibling (rare, malformed
export), we fall back to the <mathmlword> tag, then drop it silently rather
than injecting raw glyph junk.

100% FOSS: beautifulsoup4 (MIT license) + Python stdlib. No paid API calls,
no rasterization, no vision model needed for this format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from bs4 import BeautifulSoup, Tag


@dataclass
class ExtractedQuestion:
    number: Optional[str]   # e.g. "1.", "2." — None if no marker found
    text: str               # clean text with LaTeX inlined as $...$/$$...$$
    raw_html: str            # original HTML fragment, kept for audit/debug


def _inline_math(node: Tag) -> None:
    """Mutates `node` in place: replaces every math span with its LaTeX."""
    for math_span in node.find_all(
        class_=lambda c: c and ("math-inline" in c or "math-block" in c)
    ):
        latex_tag = math_span.find("latex")
        source_tag = latex_tag or math_span.find("mathmlword")

        if source_tag is None:
            math_span.decompose()
            continue

        is_block = "math-block" in (math_span.get("class") or [])
        wrapper = "$$" if is_block else "$"
        latex_str = " ".join(source_tag.get_text().split())
        math_span.replace_with(f" {wrapper}{latex_str}{wrapper} ")


def _clean_text(node: Tag) -> str:
    text = node.get_text(separator=" ")
    return " ".join(text.split())


def _top_level_question_lists(soup: BeautifulSoup) -> List[Tag]:
    """
    Returns <ul>/<ol> elements that hold questions, excluding any that are
    nested inside another question's <li> (those are sub-parts like i), ii)
    and get captured as part of the parent question's text instead).
    """
    lists = soup.find_all(["ul", "ol"], class_=["itemize", "enumerate"])
    top_level = []
    for lst in lists:
        if lst.find_parent("li") is None:
            top_level.append(lst)
    return top_level


def extract_questions_from_html(html_content: bytes | str) -> List[ExtractedQuestion]:
    """
    Main entry point. Give it raw HTML bytes/str (e.g. from an uploaded
    .html file), get back a list of clean, LLM-ready questions.

    Falls back to paragraph-level extraction if no <ul>/<ol> question
    list is found (handles plain HTML that isn't a Mathpix export).
    """
    if isinstance(html_content, bytes):
        html_content = html_content.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html_content, "html.parser")
    question_lists = _top_level_question_lists(soup)

    results: List[ExtractedQuestion] = []

    if question_lists:
        for lst in question_lists:
            for li in lst.find_all("li", recursive=False):
                li_copy = BeautifulSoup(str(li), "html.parser").find(li.name)
                marker_tag = li_copy.find(class_="li_level")
                number = marker_tag.get_text().strip() if marker_tag else None
                if marker_tag:
                    marker_tag.decompose()

                _inline_math(li_copy)
                text = _clean_text(li_copy)
                if text:
                    results.append(
                        ExtractedQuestion(number=number, text=text, raw_html=str(li))
                    )
    else:
        # Fallback: no structured list found — treat each top-level
        # paragraph/div as one candidate question.
        body = soup.find("body") or soup
        for para in body.find_all(["p", "div"], recursive=True):
            if para.find(["p", "div"]):
                continue  # skip containers, only take leaf blocks
            para_copy = BeautifulSoup(str(para), "html.parser").find(para.name)
            _inline_math(para_copy)
            text = _clean_text(para_copy)
            if text and len(text) > 3:
                results.append(
                    ExtractedQuestion(number=None, text=text, raw_html=str(para))
                )

    return results


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "rb") as f:
        qs = extract_questions_from_html(f.read())
    for q in qs:
        print(f"{q.number or '-'} | {q.text}")
