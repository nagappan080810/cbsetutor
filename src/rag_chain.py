import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from src.retriever import CBSERetriever
from src.formatter import format_answer
from rich.console import Console
import sympy

load_dotenv()
console = Console()


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a math expression EXACTLY using sympy and return the precise
    result. ALWAYS use this tool for any arithmetic, counting, algebra, or
    numeric comparison instead of computing it mentally -- including counts
    of numbers between two values, squares/cubes, differences, fractions,
    percentages, LCM/HCF, and simplification.

    Examples:
      "100**2 - 99**2 - 1"   -> counting integers strictly between 99^2 and 100^2
      "sympy.gcd(24, 36)"    -> HCF
      "sympy.lcm(24, 36)"    -> LCM
      "sympy.Rational(3,4) + sympy.Rational(1,6)"  -> fraction arithmetic

    Input must be a valid Python/sympy expression string.
    """
    try:
        result = sympy.sympify(expression, evaluate=True)
        return str(result)
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

class CBSERagChain:
    def __init__(self):
        self.retriever = CBSERetriever()

        # LLM_PROVIDER selects which backend to use without touching code.
        # Options: "deepseek" (default, direct DeepSeek API) | "nvidia" (NIM
        # hosted catalog) | "groq" (Llama via Groq).
        provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

        if provider == "nvidia":
            self.llm = ChatNVIDIA(
                model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
                api_key=os.getenv("NVIDIA_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
                top_p=float(os.getenv("LLM_TOP_P", 0.95)),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", 16384)),
                timeout=360,  # fail fast with a clear error instead of hanging
            ).bind_tools([calculate])

        elif provider == "groq":
            self.llm = ChatGroq(
                model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1,
                max_tokens=2048,
            ).bind_tools([calculate])

        else:  # "deepseek" default — direct DeepSeek API
            self.llm = ChatDeepSeek(
                model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                temperature=0.1,
                max_tokens=2048,
            ).bind_tools([calculate])

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
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    console.print(f"content generated...{chunk.content}")
                    yield chunk.content
                else:
                    console.print(f"content not generated...")
                gathered = chunk if gathered is None else gathered + chunk

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
        # likely doesn't cover this question well. Forcing the LLM to answer
        # anyway is exactly what produces hallucinated, confident-sounding
        # wrong answers. Refuse early instead.
        if not self.retriever.is_context_sufficient(results, min_confidence=35.0):
            overall = self.retriever.overall_confidence(results)
            console.print(
                f"[red]⚠ Low confidence ({overall['score']}%) — the textbook context "
                f"doesn't clearly cover this question.[/red]\n"
                f"[yellow]Showing the closest matches found, but treat the answer "
                f"with caution or rephrase the question.[/yellow]\n"
            )
            # Still proceed, but the prompt's anti-hallucination rules + this
            # warning together push toward an honest "not covered" answer
            # rather than silent confident hallucination.

        # Extract doc objects for context building
        top_docs = self.retriever.get_top_docs(results)

        # Compute overall confidence
        overall = self.retriever.overall_confidence(results)

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

        prompt = PROMPT_TEMPLATE.format(
            context=context,
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
