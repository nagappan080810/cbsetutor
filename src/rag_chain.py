import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI  # used for opencode_zen -- OpenAI-compatible endpoint, no dedicated langchain client exists
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from src.retriever import CBSERetriever
from src.formatter import format_answer
from src.latex_utils import looks_like_latex, parse_latex_expr, parse_latex_equation
from rich.console import Console

# sympy is optional: if it's unavailable, the `calculate` tool is simply not
# bound to the LLM (see CBSERagChain.__init__) and the model falls back to
# doing its own arithmetic in-line, rather than the app failing to start.
# Precise arithmetic via sympy is still strongly preferred when available --
# this is a graceful-degradation path, not the recommended normal state.
try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    sympy = None
    SYMPY_AVAILABLE = False

load_dotenv()
console = Console()

if not SYMPY_AVAILABLE:
    console.print(
        "[yellow]⚠ sympy not installed — the `calculate` tool is disabled. "
        "The LLM will compute arithmetic itself (less reliable for exact "
        "values). Run `pip install sympy` to re-enable it.[/yellow]"
    )


def _sympy_eval(expression: str):
    """
    Evaluate a sympy expression OR a short multi-line sympy script (imports,
    variable assignments, then a final expression) and return the value of
    the last statement.

    Why both paths: the LLM sometimes writes step-by-step working (assign
    intermediate variables, then combine them) rather than one inline
    expression, which plain `sympy.sympify()` cannot parse at all -- that
    mismatch was the actual cause of `calculate` erroring out on valid,
    well-formed working. `sympify` is tried first since it's the common
    case and keeps the fast/simple path fast; multi-statement exec is the
    fallback only when that fails.

    Sandboxing: exec runs with __builtins__ stripped and only `sympy` (plus
    Rational/symbols/etc. pulled to top-level for convenience) in scope --
    no file, network, or OS access is reachable from this namespace.
    """
    try:
        return sympy.sympify(expression, evaluate=True)
    except Exception:
        pass  # fall through to multi-statement exec below

    import ast

    tree = ast.parse(expression, mode="exec")
    if not tree.body:
        raise ValueError("empty expression")

    # sympy only. Earlier this allowlisted fractions/decimal/math/itertools/
    # statistics too, since production logs showed the model reaching for
    # `fractions.Fraction` -- but sympy.Rational already covers that exact
    # same use case AND composes correctly with irrational values (sqrt(6),
    # sqrt(2), etc.) in the same expression, which fractions.Fraction can't
    # do at all. Trig identity questions routinely mix both in one
    # computation, so steering the model to one consistent library (via the
    # `calculate` docstring below) is the real fix -- not widening the
    # sandbox to chase whichever stdlib module the model tries next.
    _ALLOWED_IMPORT_MODULES = {"sympy"}

    def _restricted_import(name, *args, **kwargs):
        base_module = name.split(".")[0]
        if base_module in _ALLOWED_IMPORT_MODULES:
            return __import__(name, *args, **kwargs)
        raise ImportError(f"import of '{name}' is not permitted in calculate()")

    safe_globals = {
        "__builtins__": {"__import__": _restricted_import},
        "sympy": sympy,
        **{name: getattr(sympy, name) for name in dir(sympy) if not name.startswith("_")},
    }
    safe_locals: dict = {}

    *stmts, last = tree.body
    if stmts:
        exec(compile(ast.Module(body=stmts, type_ignores=[]), "<calc>", "exec"),
             safe_globals, safe_locals)

    if isinstance(last, ast.Expr):
        # Final line is a bare expression (e.g. `result_i`) -- evaluate it
        # for its value rather than executing it as a statement.
        return eval(compile(ast.Expression(body=last.value), "<calc>", "eval"),
                    safe_globals, safe_locals)
    else:
        # Final line is itself an assignment/import/etc -- run it, then
        # report the last assigned variable if there is one, else None.
        exec(compile(ast.Module(body=[last], type_ignores=[]), "<calc>", "exec"),
             safe_globals, safe_locals)
        if isinstance(last, ast.Assign) and isinstance(last.targets[0], ast.Name):
            return safe_locals.get(last.targets[0].id)
        return None


@tool
def calculate(expression: str) -> str:
    r"""
    Evaluate a math expression or short sympy working EXACTLY and return the
    precise result. ALWAYS use this tool for any arithmetic, counting,
    algebra, or numeric comparison instead of computing it mentally --
    including counts of numbers between two values, squares/cubes,
    differences, fractions, percentages, LCM/HCF, and simplification.

    IMPORTANT for multi-part questions (i, ii, iii...): compute ALL parts in
    ONE call, not one call per part. Each separate call is a full round-trip
    to the model provider, and unnecessary round-trips are the main cause of
    slow/stalled responses on multi-part questions. Return a tuple, e.g.:
      "from sympy import Rational
      sinB = Rational(21,29); cosB = Rational(20,29)
      sinC = cosB; cosC = sinB
      part_i  = sinB*cosC + cosB*sinC
      part_ii = cosB*cosC + sinB*sinC
      (part_i, part_ii)"
    -- one call, both answers back at once.

    IMPORTANT for "verify this identity" questions: the LaTeX equation input
    above (pass the full identity including "=") handles this directly in
    one call -- no need to manually derive values step by step first.

    LATEX INPUT (usually easiest): you can pass an expression straight from
    the question text, in LaTeX, exactly as written -- e.g.
    "\frac{\cos A}{1-\tan A}+\frac{\sin A}{1-\cot A}" or, for an identity to
    verify, the FULL equation including the "=":
    "\frac{\cos A}{1-\tan A}+\frac{\sin A}{1-\cot A}=\cot A+\sin A" -- for an
    equation, this returns simplify(LHS - RHS) directly (0 means the two
    sides are equal, i.e. the identity holds; anything else means it
    doesn't). This works for ANY question's expressions, not just
    identities -- use it whenever it saves you from manually translating
    LaTeX into Python syntax, since that translation step is itself a
    common source of mistakes. Falls back automatically to the Python/sympy
    syntax below if the input isn't LaTeX-shaped.

    Only the `sympy` library is available inside this tool -- no other
    imports work (e.g. `fractions`, `decimal`, `math` will error). Use
    `sympy.Rational(a, b)` for exact fractions, NOT `fractions.Fraction` --
    Rational does the same job and, unlike Fraction, composes correctly
    with irrational values like sqrt(6) in the same expression, which is
    essential for trig identities that mix both (e.g. verifying
    sqrt(6)*tan(theta) = 2*sqrt(2)*sin(theta) exactly, not as a float
    approximation that can mask a true identity as false or vice versa).

    Accepts either:
      - a single expression: "100**2 - 99**2 - 1", "sympy.gcd(24, 36)",
        "sympy.Rational(3,4) + sympy.Rational(1,6)", "sympy.sqrt(6)/(2*sympy.sqrt(2))"
      - OR a short multi-line script with imports/assignments ending in the
        value you want, e.g.:
          "from sympy import Rational
          a = Rational(4,3)
          b = Rational(12,5)
          (a + b) / (1 - a*b)"

    If this tool returns an ERROR, do not retry the same expression in the
    same broken form more than once -- rephrase as a single self-contained
    sympy expression instead, or if it keeps failing, proceed with careful
    manual calculation and clearly show your working rather than looping.
    """
    if not SYMPY_AVAILABLE:
        return (
            "ERROR: sympy is not installed on this server, so this tool is "
            "unavailable. Compute this yourself, showing careful step-by-step "
            "working, and double-check the arithmetic before giving the final answer."
        )
    try:
        if looks_like_latex(expression):
            # Generic LaTeX path: works for whatever expression the LLM
            # pastes in, for any question type -- not tied to any specific
            # phrasing or problem pattern. If it's an equation (has '='),
            # return lhs-rhs simplified (0 means the two sides are equal,
            # which directly answers "verify this identity"-style
            # questions too, without any question-specific code).
            if "=" in expression:
                lhs, rhs = parse_latex_equation(expression)
                result = sympy.simplify(lhs - rhs)
            else:
                result = sympy.simplify(parse_latex_expr(expression))
        else:
            result = _sympy_eval(expression)
        return str(result)
    except ImportError:
        return (
            "ERROR: LaTeX parsing isn't available on this server (missing "
            "antlr4-python3-runtime). Rewrite this as plain Python/sympy "
            "syntax instead, e.g. sympy.cos(A)/(1-sympy.tan(A)) rather than "
            "\\frac{\\cos A}{1-\\tan A}."
        )
    except Exception as e:
        return f"ERROR: could not evaluate '{expression}': {e}"


PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "class_name", "subject", "language_instruction"],
    template="""You are an expert CBSE tutor for {class_name}, subject: {subject}.

{language_instruction}

Use ONLY the context below from NCERT textbooks to answer.

CRITICAL ANTI-HALLUCINATION RULES — follow these strictly:
- Base your ENTIRE answer only on facts present in the CONTEXT below.
- Do NOT add facts, figures, dates, formulas, or examples that are not in the CONTEXT,
  even if you recall them from general knowledge — the textbook's exact wording and
  values may differ and a wrong "confident-sounding" answer is worse than admitting
  the context is incomplete.
- If the CONTEXT only partially covers the question, answer the part it covers and
  explicitly say: "The provided textbook context does not cover [the missing part]."
- If the CONTEXT does not contain the answer at all, say so clearly in the same
  language instead of guessing — do not invent an answer.
- Do not blend information from outside the CBSE/NCERT context, even if it seems
  more complete or commonly known.

Rules:
- For maths: show every step numbered (Step 1, Step 2...), using only the
  method/formula shown in the context
- For ANY arithmetic, counting, or numeric result (additions, squares,
  differences, fractions, percentages, LCM/HCF, counting integers in a
  range, etc.) — NEVER compute it mentally. ALWAYS call the `calculate`
  tool with the exact expression and use its returned result as the final
  number in your answer.
- Counting trap to watch for: "numbers BETWEEN a and b" means STRICTLY
  EXCLUDING both a and b. The count is (b - a - 1), not (b - a). Always
  express this as a `calculate` call, e.g. for "numbers between 99^2 and
  100^2" call calculate("100**2 - 99**2 - 1"), never compute it by hand.
- For languages (Hindi/Kannada/Tamil/Sanskrit): explain grammar rules clearly,
  quoting the exact rule text from context where possible
- Define formulas or terms before using them — using the context's own definition
- Use simple language for a school student
- End with a short summary or key takeaway drawn only from the context
- Use markdown formatting (bold for key terms, code blocks for formulas/equations)

---
CONTEXT:
{context}

---
QUESTION: {question}

ANSWER:"""
)

# ── Fallback prompt: used only when retrieval confidence is too low to
# ground the answer (see is_context_sufficient() gate in ask()/api.py).
# Forcing PROMPT_TEMPLATE's strict "ONLY use context" rules onto irrelevant
# or empty context produces either a flat refusal or, worse, an LLM that
# keeps reaching for the `calculate` tool trying to reconcile numbers that
# aren't actually in the context -- which is what caused the retry-stall
# seen on self-contained problems (e.g. "verify this trig identity") that
# have no matching textbook passage to retrieve, by design. This prompt
# instead lets the model answer from its own subject knowledge, clearly
# labelled as such, bounded to the CBSE/NCERT syllabus for the given class.
GENERAL_KNOWLEDGE_PROMPT = PromptTemplate(
    input_variables=["question", "class_name", "subject", "language_instruction"],
    template="""You are an expert CBSE tutor for {class_name}, subject: {subject}.

{language_instruction}

No sufficiently relevant passage was found in the NCERT textbook corpus for
this question -- this is expected for self-contained problems (e.g. "verify
this identity", "solve for x") that don't correspond to any specific
textbook passage, not necessarily a sign something is wrong.

Answer using your own subject knowledge instead, following these rules:
- Stay strictly within the CBSE {class_name} syllabus for {subject} -- do not
  use methods, notation, or content from a different grade level.
- Show every step numbered (Step 1, Step 2...).
- For ANY arithmetic, algebra, or numeric result, ALWAYS call the
  `calculate` tool with a single self-contained expression and use its
  returned result as the final number -- never compute it mentally. If a
  calculate call errors, do not retry the same broken form; rephrase once
  as a single expression, or otherwise proceed with careful manual working.
- Use simple language for a school student.
- End with a short summary or key takeaway.
- Use markdown formatting (bold for key terms, code blocks for formulas/equations).

---
QUESTION: {question}

ANSWER:"""
)

class CBSERagChain:
    def __init__(self):
        self.retriever = CBSERetriever()

        # LLM_PROVIDER selects which backend to use without touching code.
        # Options: "deepseek" (default, direct DeepSeek API) | "nvidia" (NIM
        # hosted catalog) | "groq" (Llama via Groq) | "opencode_zen" (OpenAI-
        # compatible gateway, currently free-tier DeepSeek V4 Flash -- see
        # notes below on why this is dev/experimentation-only, not prod).
        provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

        if provider == "nvidia":
            self.llm_raw = ChatNVIDIA(
                model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
                api_key=os.getenv("NVIDIA_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
                top_p=float(os.getenv("LLM_TOP_P", 0.95)),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", 16384)),
                timeout=360,  # fail fast with a clear error instead of hanging
            )

        elif provider == "groq":
            self.llm_raw = ChatGroq(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1,
                max_tokens=2048,
            )

        elif provider == "opencode_zen":
            # No dedicated langchain_opencode client exists -- Zen exposes a
            # generic OpenAI-compatible /chat/completions endpoint, so we
            # point the generic ChatOpenAI client at it via base_url.
            # NOTE: the free-tier model listing here (deepseek-v4-flash-free)
            # is promotional and carries a data-retention exception during
            # its free period -- use for dev/experimentation, not production
            # worksheet traffic. Prefer "deepseek" (first-party API) for prod.
            self.llm_raw = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "deepseek-v4-flash-free"),
                api_key=os.getenv("OPENCODE_API_KEY"),
                base_url="https://opencode.ai/zen/v1",
                temperature=0.1,
                max_tokens=2048,
            )

        else:  # "deepseek" default — direct DeepSeek API
            self.llm_raw = ChatDeepSeek(
                model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                temperature=0.1,
                max_tokens=2048,
            )

        # self.llm_raw has no tools bound -- used wherever a turn MUST return
        # clean text/JSON with no possibility of a tool-call turn (structured
        # extraction, e.g. concept_graph.py's extraction pass). self.llm is
        # the tool-bound variant used for normal Q&A/worksheet generation
        # where the `calculate` tool may legitimately be invoked mid-answer.
        # Tool binding itself is skipped entirely when sympy isn't installed
        # -- an unbound model can't be asked to call a tool that doesn't
        # exist, so there's nothing to bind in that case.
        self.llm = self.llm_raw.bind_tools([calculate]) if SYMPY_AVAILABLE else self.llm_raw

    async def astream_with_tools(self, messages, max_tool_iterations: int = 4):
        """
        Async token-streaming generator that transparently handles
        `calculate` tool calls mid-stream. Shared by CLI (ask()) callers
        that want streaming and by api.py's SSE endpoints, so the
        provider-selection logic AND the tool-calling behaviour live in
        exactly one place instead of being duplicated per caller.

        Yields plain text chunks (str) as they arrive. Tool-call requests
        from the model are executed locally and never yielded as visible
        tokens -- only the model's follow-up text is streamed to the caller.
        """
        iterations = 0
        while True:
            gathered = None
            chunk_count = 0
            async for chunk in self.llm.astream(messages):
                chunk_count += 1
                if chunk.content:
                    console.print(f"content generated...{chunk.content}")
                    yield chunk.content
                else:
                    console.print(f"content not generated...")
                gathered = chunk if gathered is None else gathered + chunk

            if chunk_count == 0:
                # astream() completed its iteration having yielded nothing
                # at all -- not a normal empty-content chunk, but zero
                # chunks total. This is how a provider-side failure (rate
                # limit, empty SSE body, etc.) surfaces when the client
                # library swallows it instead of raising. Left unchecked,
                # this silently ends the generator, callers see an empty
                # full_response, and any downstream json.loads() fails with
                # a misleading "Expecting value: line 1 column 1" instead
                # of the actual cause. Raise here, at the call site, so it
                # propagates as a real error to whichever caller is
                # wrapping this in try/except (api.py's SSE handlers).
                raise RuntimeError(
                    f"LLM provider returned an empty stream (no chunks received). "
                    f"This usually means a rate-limit or provider-side error that "
                    f"didn't surface as an exception. LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'unknown')!r}."
                )

            tool_calls = getattr(gathered, "tool_calls", None) if gathered else None

            if not tool_calls:
                break

            # Always execute the tool calls the model just requested and
            # feed the results back in, regardless of the iteration cap --
            # previously, hitting the cap while tool_calls was still truthy
            # caused an immediate `break` here, discarding this turn
            # entirely. That meant zero visible text was ever yielded for
            # questions needing >= max_tool_iterations calculate() calls
            # (e.g. multi-step physics problems), even though the model
            # never got a chance to write its final answer.
            messages.append(gathered)
            for tool_call in tool_calls:
                if tool_call["name"] == "calculate":
                    expr        = tool_call["args"].get("expression", "")
                    tool_result = calculate.invoke({"expression": expr})
                    console.print(f"[dim]  🧮 calculate({expr!r}) = {tool_result}[/dim]")
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                    )

            if iterations >= max_tool_iterations:
                # Force one final, tool-free turn so the model always
                # produces visible output instead of silently ending here.
                # Deliberately format-neutral -- this same method backs both
                # free-text Q&A (api.py's /api/ask) and strict-JSON worksheet
                # generation (api.py's worksheet_stream), so it must not tell
                # the model to switch to "plain text" when it may actually be
                # mid-way through a JSON object it still needs to close out.
                messages.append(HumanMessage(
                    content="You now have all the tool results you need. "
                            "Continue and complete your response now, in "
                            "exactly the format you were already asked to "
                            "use -- do not call any more tools."
                ))
                async for chunk in self.llm.astream(messages):
                    if chunk.content:
                        console.print(f"content generated (final turn)...{chunk.content}")
                        yield chunk.content
                break

            iterations += 1

    def ask(
        self,
        question:             str,
        class_name:           str,
        subject:              str,
        language_instruction: str,
        difficulty:            str  = None,   # optional: "easy"|"medium"|"hard"|"mixed"
        prefer_exercise:        bool = None   # None = auto-detect from question text
    ):
        # Auto-detect intent to retrieve exercise/Figure-It-Out content if
        # not explicitly specified — covers the common case where a student
        # just asks "give me Figure It Out questions" or "hard questions
        # from chapter 3" without the caller setting the flag manually.
        if prefer_exercise is None:
            ql = question.lower()
            prefer_exercise = any(kw in ql for kw in [
                "figure it out", "exercise", "hard question", "difficult question",
                "harder question", "tough question", "challenging question",
                "higher order", "hots", "activity question", "practice question"
            ])

        console.print(
            f"\n[bold yellow]🔍 Searching {class_name} → {subject}...[/bold yellow]"
        )

        # retrieve_and_rerank now returns list of dicts with doc + confidence,
        # and is metadata-aware so harder / activity-based content can surface
        results = self.retriever.retrieve_and_rerank(
            question,
            class_filter=class_name,
            subject_filter=subject,
            difficulty=difficulty,
            prefer_exercise=prefer_exercise
        )

        if not results:
            console.print(
                "[red]No content found for this class/subject. "
                "Make sure PDFs are placed in the correct folder and ingested.[/red]"
            )
            return

        # ── Anti-hallucination gate ─────────────────────────────────────────
        # If even the best-matching chunk has low confidence, the textbook
        # likely doesn't cover this question well -- OR the question is
        # simply self-contained (e.g. "verify this identity") and was never
        # going to match a specific textbook passage in the first place.
        # Rather than forcing the strict context-only prompt onto irrelevant
        # chunks (which produces either a refusal or a confused answer that
        # tries to reconcile numbers not actually present), fall back to
        # letting the LLM answer from its own CBSE-syllabus knowledge,
        # clearly labelled as ungrounded.
        context_sufficient = self.retriever.is_context_sufficient(results, min_confidence=35.0)
        overall = self.retriever.overall_confidence(results)

        if not context_sufficient:
            console.print(
                f"[yellow]⚠ Low confidence ({overall['score']}%) — no closely matching "
                f"textbook passage found. Falling back to a general-knowledge answer "
                f"(clearly labelled), instead of forcing a weak/irrelevant context.[/yellow]\n"
            )

        # Extract doc objects for context building
        top_docs = self.retriever.get_top_docs(results)

        # Limit context size — raised from 2000 to 6000 chars. The previous
        # 2000-char cap usually fit only 1-2 of the 5 re-ranked chunks, so
        # the LLM was working from a sliver of the chapter and filled the
        # rest in from general knowledge (i.e. hallucinated). 6000 chars
        # comfortably fits all 5 chunks while staying well within context.
        MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 6000))
        context_parts = []
        total_len     = 0
        for doc in top_docs:
            if total_len + len(doc.page_content) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(doc.page_content)
            total_len += len(doc.page_content)

        context = "\n\n---\n\n".join(context_parts)

        if context_sufficient:
            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=question,
                class_name=class_name.replace("_", " ").title(),
                subject=subject.replace("_", " ").title(),
                language_instruction=language_instruction
            )
        else:
            prompt = GENERAL_KNOWLEDGE_PROMPT.format(
                question=question,
                class_name=class_name.replace("_", " ").title(),
                subject=subject.replace("_", " ").title(),
                language_instruction=language_instruction
            )

        console.print("[bold yellow]🤖 Generating answer...[/bold yellow]\n")

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)

            # Tool-calling loop — if the model requests calculate(...),
            # execute it locally with sympy and feed the exact result back
            # in, rather than trusting the model's own mental arithmetic.
            # This is what fixes off-by-one errors like "numbers between
            # 99^2 and 100^2" (correct answer 198, not 199).
            max_tool_iterations = 3
            iterations = 0
            while getattr(response, "tool_calls", None) and iterations < max_tool_iterations:
                messages.append(response)
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "calculate":
                        expr        = tool_call["args"].get("expression", "")
                        tool_result = calculate.invoke({"expression": expr})
                        console.print(
                            f"[dim]  🧮 calculate({expr!r}) = {tool_result}[/dim]"
                        )
                        messages.append(
                            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                        )
                response   = self.llm.invoke(messages)
                iterations += 1

            answer = response.content or "(No answer text returned by the model.)"
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


# ── Module-level singleton ───────────────────────────────────────────────────
# Lazy so importing this module (e.g. from api.py) doesn't immediately try to
# build embeddings/vectorstore/LLM clients at import time.
_rag_chain = None

def get_rag_chain() -> "CBSERagChain":
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = CBSERagChain()
    return _rag_chain
