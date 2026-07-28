"""
Standalone extractor + solver for HTML question files (Mathpix-style exports).

Runs independently of api.py / the RAG retriever — appropriate here because
these questions (e.g. trig identities) are self-contained: solving them needs
reasoning, not NCERT chapter grounding. If you later want answers grounded in
a specific chapter's method/notation, route through the normal /api/ask
pipeline instead.

FOSS-first provider order:
  1. ollama   - local, free, no API key (DEFAULT). Requires `ollama serve`
               running and a model pulled, e.g. `ollama pull llama3.1`.
  2. groq     - free tier, hosted, OpenAI-compatible. Needs GROQ_API_KEY.
  3. deepseek - paid (cheap), hosted, OpenAI-compatible. Needs DEEPSEEK_API_KEY.
  4. nvidia   - NVIDIA NIM, free tier available, hosted. Needs NVIDIA_API_KEY.

Select via LLM_PROVIDER env var (matches your existing api.py convention).
Defaults to ollama if unset, since that's the no-cost, no-account option.

Usage:
    python3 standalone_solve.py path/to/file.html
    python3 standalone_solve.py path/to/file.html --provider groq
    python3 standalone_solve.py path/to/file.html --json out.json
    python3 standalone_solve.py path/to/file.html --extract-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Optional

from html_question_extractor import ExtractedQuestion, extract_questions_from_html

SOLVE_SYSTEM_PROMPT = (
    "You are a CBSE/NCERT maths tutor for classes 8-10. Solve the given "
    "question fully and rigorously. Show working step by step, then end "
    "with a line starting exactly 'Final Answer:' followed by the result. "
    "Use LaTeX ($...$) for math in your response. Do not use content beyond "
    "the CBSE class 8-10 syllabus."
)

PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model": os.environ.get("OLLAMA_MODEL", "llama3.1"),
        "api_key": "ollama",  # unused, Ollama's OpenAI-compat endpoint ignores it
        "cost_note": "Local, free, no rate limits. Requires `ollama serve` + a pulled model.",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "api_key": os.environ.get("GROQ_API_KEY"),
        "cost_note": "Hosted, free tier with rate limits. Needs GROQ_API_KEY.",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "cost_note": "Hosted, paid (low cost per token). Needs DEEPSEEK_API_KEY.",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
        "api_key": os.environ.get("NVIDIA_API_KEY"),
        "cost_note": "Hosted, free tier available. Needs NVIDIA_API_KEY.",
    },
}


@dataclass
class SolvedQuestion:
    number: Optional[str]
    question: str
    answer: str
    provider: str


def _chat_completion(provider: str, question_text: str, timeout: int = 45) -> str:
    cfg = PROVIDER_DEFAULTS[provider]
    if provider != "ollama" and not cfg["api_key"]:
        raise RuntimeError(
            f"{provider} selected but its API key env var is not set. "
            f"Set the key, or use --provider ollama for a free local option."
        )

    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
            {"role": "user", "content": question_text},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    req = urllib.request.Request(
        url=f"{cfg['base_url']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach {provider} at {cfg['base_url']} ({e}). "
            f"If using ollama, make sure `ollama serve` is running."
        ) from e

    return body["choices"][0]["message"]["content"].strip()


def solve_questions(
    questions: list[ExtractedQuestion], provider: str
) -> list[SolvedQuestion]:
    solved = []
    for q in questions:
        try:
            answer = _chat_completion(provider, q.text)
        except RuntimeError as e:
            answer = f"[ERROR solving this question: {e}]"
        solved.append(
            SolvedQuestion(number=q.number, question=q.text, answer=answer, provider=provider)
        )
    return solved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html_file", help="Path to the HTML question file")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDER_DEFAULTS.keys()),
        default=os.environ.get("LLM_PROVIDER", "ollama"),
        help="LLM provider to use for solving (default: ollama, local & free)",
    )
    parser.add_argument("--extract-only", action="store_true", help="Only extract questions, skip solving")
    parser.add_argument("--json", metavar="PATH", help="Write results as JSON to this path")
    args = parser.parse_args()

    with open(args.html_file, "rb") as f:
        questions = extract_questions_from_html(f.read())

    if not questions:
        print("No questions found in file.", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(questions)} question(s).\n")

    if args.extract_only:
        for q in questions:
            print(f"{q.number or '-'} | {q.text}\n")
        if args.json:
            with open(args.json, "w") as f:
                json.dump([asdict(q) for q in questions], f, indent=2)
        return

    cfg = PROVIDER_DEFAULTS[args.provider]
    print(f"Solving with provider={args.provider} model={cfg['model']} ({cfg['cost_note']})\n")

    solved = solve_questions(questions, args.provider)
    for s in solved:
        print(f"--- Question {s.number or ''} ---")
        print(s.question)
        print()
        print(s.answer)
        print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(s) for s in solved], f, indent=2)
        print(f"Wrote results to {args.json}")


if __name__ == "__main__":
    main()
