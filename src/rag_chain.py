import os
import re
import time
import asyncio
import requests
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, AIMessageChunk
from langchain_core.tools import tool
from src.retriever import CBSERetriever
from src.formatter import format_answer
from rich.console import Console
import sympy

load_dotenv()
console = Console()


class CharitraCloudChat:
    """
    Thin client for the already-deployed Charitra HF Space endpoint
    (https://lijoraju-charitra-backend.hf.space/query) — see
    customizedmodel.txt. Chosen deliberately over pulling the GGUF model
    locally: this hits the model in the cloud, nothing to download.

    IMPORTANT — what this bypasses (by design, per user's choice):
      - This endpoint does ITS OWN retrieval internally, over its OWN FAISS
        index built only over NCERT Class 10 Social Science. It does not
        accept a context override.
      - Using it as LLM_PROVIDER means this app's own Chroma retriever,
        class/subject filters, re-ranking, and metadata boosts are NOT used
        to build the answer — we just forward the raw question and return
        whatever this endpoint says.
      - Answers will only ever reflect NCERT Class 10 Social Science
        content, regardless of which class/subject the user selects in
        this app's UI.
      - It's single-shot, not token-streamed. astream() below fakes
        streaming by yielding the whole answer as one chunk, so it stays
        compatible with astream_with_tools() without changing call sites.
      - The `calculate` tool is never invoked for this provider — the
        endpoint has no concept of tool calls, and we don't bind any tools
        to this class.
      - It CANNOT produce the structured multi-question JSON a worksheet
        needs (it returns a single free-text answer to a factual
        question). Worksheet generation is explicitly blocked for this
        provider in api.py rather than being attempted and failing.
    """
    def __init__(self, base_url: str = None, top_k: int = 3, timeout: int = 60):
        self.base_url = (base_url or os.getenv(
            "CHARITRA_API_URL", "https://lijoraju-charitra-backend.hf.space"
        )).rstrip("/")
        self.top_k   = top_k
        self.timeout = timeout

    @staticmethod
    def _extract_question(messages) -> str:
        """
        Our own prompt templates wrap the user's actual question inside a
        large instructions+context blob ending in
        'QUESTION: <question>\\n\\nANSWER:'. This endpoint expects a plain
        question, so pull just that part back out. Falls back to the raw
        message content if the marker isn't found (e.g. called directly).
        """
        raw = ""
        for m in reversed(messages):
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content")
            if content:
                raw = content
                break
        match = re.search(r"QUESTION:\s*(.*?)\s*ANSWER:", raw, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else raw.strip()

    def invoke(self, messages):
        question = self._extract_question(messages)
        last_err = None
        # Two distinct failure modes from a free-tier HF Space:
        #   1. Read timeout -> Space was asleep, woke up mid-request. One
        #      retry after it's had time to finish loading usually works.
        #   2. HTTP 503 -> Space container is unhealthy: still booting,
        #      crashed, OOM'd, or (on some tiers) fully paused. A paused
        #      Space needs the OWNER to restart it from the HF dashboard --
        #      no client-side retry fixes that. We still retry a couple of
        #      times with backoff in case it's mid-boot, but don't pretend
        #      retries can fix a genuinely down/paused Space.
        delays = [5, 15]  # seconds between attempts 1->2 and 2->3
        for attempt in range(len(delays) + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/query",
                    json={"query": question, "top_k": self.top_k},
                    timeout=self.timeout,
                )
                if resp.status_code == 503:
                    last_err = requests.HTTPError(
                        "503 Service Unavailable (Space booting, crashed, or paused)"
                    )
                    if attempt < len(delays):
                        time.sleep(delays[attempt])
                    continue
                resp.raise_for_status()
                answer = resp.json().get("response", "(No response field returned by Charitra API.)")
                return AIMessage(content=answer)
            except requests.Timeout as e:
                last_err = e
                continue  # likely a cold start -- try again
            except requests.RequestException as e:
                last_err = e
                break  # non-timeout, non-503 errors (4xx, connection refused) won't fix themselves

        return AIMessage(content=(
            f"(Charitra API unavailable after retries: {last_err}. This is a "
            f"third-party public demo Space ({self.base_url}) -- if it's paused "
            f"or crashed, only its owner can restart it. Check its live status "
            f"at https://huggingface.co/spaces/lijoraju/charitra-backend before "
            f"retrying further.)"
        ))

    async def astream(self, messages):
        loop = asyncio.get_event_loop()
        msg  = await loop.run_in_executor(None, self.invoke, messages)
        yield AIMessageChunk(content=msg.content)


class TinyLlamaLocalChat:
    """
    Wraps ChatLlamaCpp to fix a prompt-format mismatch, not just wrap the
    model directly.

    This app's PROMPT_TEMPLATE / worksheet prompts are written for large
    instruction-following models (DeepSeek/NVIDIA/Groq) -- CBSE-tutor
    persona, anti-hallucination rules, step-numbering, calculate-tool
    instructions, etc. But per customizedmodel.txt, this TinyLlama-1.1B
    checkpoint was fine-tuned on a much simpler format:

        Context:<chunk1><chunk2><chunk3>
        Question: <question>

    A 1.1B model has very little capacity to generalize to a prompt shape
    it never saw during fine-tuning -- sent the elaborate CBSE prompt as-is,
    it mostly ignores the instructions and falls back to its own training
    habits, producing a looser, "approximate" answer instead of a precise,
    context-grounded one. Rewriting the prompt to match its actual
    fine-tuning format before it ever reaches the model is the single
    biggest lever for getting closer to the exact/grounded answers this
    checkpoint is actually capable of.
    """
    def __init__(self, chat_llama_cpp):
        self._llm = chat_llama_cpp

    @staticmethod
    def _rewrite(messages):
        raw = ""
        for m in reversed(messages):
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content")
            if content:
                raw = content
                break

        ctx_match = re.search(r"CONTEXT:\s*(.*?)\s*---\s*QUESTION:", raw, re.DOTALL | re.IGNORECASE)
        q_match   = re.search(r"QUESTION:\s*(.*?)\s*ANSWER:", raw, re.DOTALL | re.IGNORECASE)

        if ctx_match and q_match:
            context  = ctx_match.group(1).strip()
            question = q_match.group(1).strip()
            simplified = f"Context:{context}\nQuestion: {question}\n"
            return [HumanMessage(content=simplified)]

        # Couldn't find the CONTEXT:/QUESTION: markers (e.g. worksheet
        # prompts, which have a different shape) -- pass through
        # unchanged. Worksheet generation is blocked for this provider in
        # api.py anyway, so this path is mainly a safety net.
        return messages

    def invoke(self, messages):
        return self._llm.invoke(self._rewrite(messages))

    async def astream(self, messages):
        async for chunk in self._llm.astream(self._rewrite(messages)):
            yield chunk


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
        # hosted catalog) | "groq" (Llama via Groq) | "customized" (calls
        # the deployed Charitra HF Space endpoint in the cloud — see
        # CharitraCloudChat above for what this bypasses) | "customized_local"
        # (loads the same fine-tuned TinyLlama GGUF locally via
        # llama-cpp-python — no network dependency, no cloud Space uptime
        # risk, but you own the download + the CPU inference cost).
        provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

        if provider == "customized_local":
            # Local fine-tuned model (see customizedmodel.txt) — runs fully
            # offline via llama-cpp-python, no API key or network call
            # needed once the file is on disk.
            #
            # NOTE ON TOOL CALLING: TinyLlama-1.1B is NOT a reliable tool-
            # calling model. We deliberately do NOT call .bind_tools() here
            # — astream_with_tools() and ask() already handle "no
            # tool_calls returned" gracefully, so nothing breaks, but
            # expect weaker arithmetic accuracy (numbered step-by-step
            # math, "numbers between X and Y" counting, etc.) on this
            # provider compared to deepseek/nvidia/groq.
            from langchain_community.chat_models import ChatLlamaCpp

            model_path = os.getenv("CUSTOM_MODEL_PATH", "models/tinyllama-merged.gguf")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"CUSTOM_MODEL_PATH not found: {model_path!r}. Run "
                    f"download_model.py (or the huggingface-cli command in "
                    f"its docstring) to fetch tinyllama-merged.gguf from "
                    f"https://huggingface.co/lijoraju/edurag-model, or set "
                    f"CUSTOM_MODEL_PATH to wherever you already saved it."
                )

            self.llm = TinyLlamaLocalChat(ChatLlamaCpp(
                model_path=model_path,
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.0)),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", 1024)),
                n_ctx=int(os.getenv("LLM_N_CTX", 4096)),
                n_gpu_layers=int(os.getenv("LLM_GPU_LAYERS", 0)),  # 0 = CPU only
                n_batch=int(os.getenv("LLM_N_BATCH", 256)),
                verbose=False,
            ))

        elif provider == "customized":
            self.llm = CharitraCloudChat(
                base_url=os.getenv("CHARITRA_API_URL"),
                top_k=int(os.getenv("CHARITRA_TOP_K", 3)),
                timeout=int(os.getenv("CHARITRA_TIMEOUT", 180)),
            )
            # No .bind_tools() — this endpoint has no concept of tool calls.

        elif provider == "nvidia":
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

    async def astream_with_tools(self, messages, max_tool_iterations: int = 3):
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
            if not tool_calls or iterations >= max_tool_iterations:
                break

            messages.append(gathered)
            for tool_call in tool_calls:
                if tool_call["name"] == "calculate":
                    expr        = tool_call["args"].get("expression", "")
                    tool_result = calculate.invoke({"expression": expr})
                    console.print(f"[dim]  🧮 calculate({expr!r}) = {tool_result}[/dim]")
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                    )
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
