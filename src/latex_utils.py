r"""
Shared LaTeX-cleanup + parsing helpers, built on sympy's LaTeX parser
(sympy.parsing.latex, backed by antlr4-python3-runtime).

Extracted into its own module so this logic has exactly one home rather
than being duplicated anywhere it's needed -- the `calculate` tool in
rag_chain.py is the current (and intended to be the only) caller.

These fixes were found by testing against REAL extracted CBSE questions,
not written speculatively -- each one corresponds to a confirmed parse
failure or silent misparse:
  - \mathrm{~A} (Mathpix's roman-font + spacing-tilde wrapper) crashes the
    parser outright (LexerNoViableAltException).
  - bare \mathrm{A} is misread as an unknown symbol "mathrm" implicitly
    multiplied by A, rather than being unwrapped to plain A.
  - \sin^{n}\theta-style notation (near-universal in trig textbooks) is
    silently misparsed as "sin of (theta raised to something else
    entirely)" rather than "(sin theta) to the n" -- this is the most
    dangerous case since it doesn't error, it just gives a WRONG
    expression back with no indication anything went wrong.
  - \operatorname{cosec}/\cosec isn't standard LaTeX and needs normalizing
    to \csc before the parser will accept it.
"""

from __future__ import annotations

import re

from sympy.parsing.latex import parse_latex

_TRIG_FNS = ["sin", "cos", "tan", "cot", "sec", "csc", "cosec"]
_TRIG_FN_ALIASES = {"cosec": "csc"}

_POWER_NOTATION_RE = re.compile(
    r"\\(" + "|".join(_TRIG_FNS) + r")\s*\^\s*\{?(\d+)\}?\s*"
    r"(\\[a-zA-Z]+|[A-Za-z])"
)


def looks_like_latex(expression: str) -> bool:
    """
    Heuristic: does this look like LaTeX (as opposed to Python/sympy
    syntax)? Checked before deciding which parsing path to use in the
    `calculate` tool. Deliberately conservative -- false negatives just
    mean it falls through to the existing Python-syntax path, which is
    harmless; false positives would send valid Python through the LaTeX
    parser and fail, which is more disruptive.
    """
    stripped = expression.strip()
    if stripped.startswith("$") or "\\frac" in stripped or "\\sqrt" in stripped:
        return True
    # A backslash followed by a letter (a LaTeX command) that ISN'T also
    # valid Python (Python has no such syntax at all) is a strong signal.
    return bool(re.search(r"\\[a-zA-Z]", stripped))


def clean_latex(latex: str) -> str:
    """Apply all known fixes for real-world parse failures, in order."""
    latex = latex.strip().strip("$")
    latex = re.sub(r"\\mathrm\{~?([^}]*)\}", r"\1", latex)
    latex = latex.replace("~", " ")
    latex = _fix_trig_power_notation(latex)
    latex = re.sub(r"\\operatorname\{cosec\}", r"\\csc", latex)
    latex = latex.replace(r"\cosec", r"\csc")
    return latex


def _fix_trig_power_notation(latex: str) -> str:
    """Rewrite \\sin^{n}X -> (\\sin X)^{n} -- see module docstring."""
    def repl(m):
        fn, power, arg = m.group(1), m.group(2), m.group(3)
        fn = _TRIG_FN_ALIASES.get(fn, fn)
        return f"(\\{fn} {arg})^{{{power}}}"
    return _POWER_NOTATION_RE.sub(repl, latex)


def parse_latex_expr(latex: str):
    """Clean + parse a single LaTeX expression (no '=') into a sympy expr."""
    return parse_latex(clean_latex(latex))


def parse_latex_equation(latex: str):
    """
    Clean + parse a LaTeX equation (containing '=') into a
    (lhs_expr, rhs_expr) tuple of sympy expressions.
    """
    cleaned = clean_latex(latex)
    if "=" not in cleaned:
        raise ValueError("not an equation (no '=' found)")
    lhs_latex, rhs_latex = cleaned.split("=", 1)
    return parse_latex(lhs_latex), parse_latex(rhs_latex)
