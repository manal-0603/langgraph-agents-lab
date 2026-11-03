"""Intelligent University Assistant — capstone project of the LangGraph lab.

Stand-alone version of PART X of ``langgraph_lab.ipynb``.

Architecture
------------
User question -> classifier -> one of five specialised agents
             -> (optional) tools -> back to the same agent -> final answer
Empty input or API error -> fallback node.

The model is a free one served through OpenRouter (OpenAI-compatible API).

Usage
-----
    export OPENROUTER_API_KEY="sk-or-..."
    export OPENROUTER_MODEL="nvidia/nemotron-3-ultra-550b-a55b:free"   # optional, this is the default
    python src/university_assistant.py            # interactive chat
    python src/university_assistant.py --demo     # run the demo queries
    python src/university_assistant.py --mermaid  # print the graph as Mermaid
"""

from __future__ import annotations

import ast
import operator as op
import os
import sys
from datetime import datetime
from getpass import getpass
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --------------------------------------------------------------------------- #
# 1. Tools
# --------------------------------------------------------------------------- #

_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _safe_eval(node: ast.AST):
    """Evaluate an arithmetic AST. Anything that is not a number or an
    arithmetic operator raises ValueError, so no code can be executed."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression such as '25 * 4 + 10'. Supports + - * / ** % //."""
    try:
        return str(_safe_eval(ast.parse(expression.strip(), mode="eval")))
    except Exception:
        return "Invalid expression"


@tool
def current_datetime(_: str = "") -> str:
    """Return the current date and time. Use for any question about today's date or the current time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


TOOLS = [calculator, current_datetime]

# --------------------------------------------------------------------------- #
# 2. State, categories and prompts
# --------------------------------------------------------------------------- #

CATEGORIES = ["COURSES", "PROGRAMMING", "MATHEMATICS", "ADMINISTRATION", "GENERAL"]
NODE_NAMES = {c: f"{c.lower()}_agent" for c in CATEGORIES}

SYSTEM_PROMPTS = {
    "COURSES": "You are a university courses advisor. Explain course content, prerequisites, "
               "schedules and study strategies. If you lack institution-specific data, say so honestly.",
    "PROGRAMMING": "You are a programming tutor. Give clear explanations and correct Python code when appropriate.",
    "MATHEMATICS": "You are a mathematics tutor. Show steps. ALWAYS use the calculator tool for arithmetic.",
    "ADMINISTRATION": "You are a university administration assistant (enrollment, fees, certificates, procedures). "
                      "Never invent official rules or deadlines; recommend contacting the administration office when unsure.",
    "GENERAL": "You are a helpful university assistant. Use the date/time tool for questions about the current date or time.",
}


class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]  # reducer: append, do not overwrite
    category: str


# --------------------------------------------------------------------------- #
# 3. Graph construction
# --------------------------------------------------------------------------- #

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free models on OpenRouter that support tool calling (October 2026), best first.
# If the first one is overloaded or rate-limited (HTTP 429), set OPENROUTER_MODEL or pick the next one.
FREE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",   # largest: 550B MoE, built for agent orchestration
    "thinkingmachines/inkling:free",            # 975B MoE, general reasoning + tool use
    "google/gemma-4-31b-it:free",               # lighter, less often saturated
    "google/gemma-4-26b-a4b-it:free",
]
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", FREE_MODELS[0])


def build_assistant(model: str = DEFAULT_MODEL, temperature: float = 0.0):
    """Build and compile the university assistant graph.

    The model is served by OpenRouter (OpenAI-compatible API), so ``ChatOpenAI``
    is reused with a different ``base_url`` and the ``OPENROUTER_API_KEY``.
    """
    # Free models are sometimes overloaded (503) or rate-limited (429): OpenRouter's
    # "models" parameter retries the request on the next model of the list automatically.
    fallbacks = [m for m in FREE_MODELS if m != model][:2]   # OpenRouter accepts at most 3 models per request
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=3,
        extra_body={"models": [model] + fallbacks},
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    # ---- nodes -------------------------------------------------------------
    def classify(state: AssistantState):
        last = state["messages"][-1]
        text = (last.content if hasattr(last, "content") else str(last)).strip()
        if not text:                                   # error handling: empty question
            return {"category": "EMPTY"}
        prompt = f"""
        Classify the student's question into exactly one category:
        COURSES (course content, syllabus, schedules, exams, grading),
        PROGRAMMING, MATHEMATICS,
        ADMINISTRATION (enrollment, fees, certificates, procedures, deadlines),
        GENERAL (anything else).

        Question: {text}

        Return only the category name.
        """
        try:
            raw = llm.invoke(prompt).content.strip().upper()
        except Exception as e:                         # error handling: API errors
            print("Classifier API error:", e, file=sys.stderr)
            return {"category": "ERROR"}
        return {"category": next((c for c in CATEGORIES if c in raw), "GENERAL")}

    def make_agent(category: str):
        system = SystemMessage(content=SYSTEM_PROMPTS[category])

        def node(state: AssistantState):
            try:
                return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}
            except Exception as e:                     # error handling: API errors
                return {"messages": [AIMessage(
                    content=f"Sorry, I ran into a technical problem while answering "
                            f"({type(e).__name__}). Please try again.")]}
        node.__name__ = NODE_NAMES[category]
        return node

    def fallback(state: AssistantState):
        if state["category"] == "EMPTY":
            msg = "Your question seems to be empty. Please type a question and I'll help."
        else:
            msg = "Sorry, I couldn't process your question right now. Please try again in a moment."
        return {"messages": [AIMessage(content=msg)]}

    # ---- routing functions ---------------------------------------------------
    def route_after_classifier(state: AssistantState):
        return NODE_NAMES.get(state["category"], "fallback")

    def route_after_agent(state: AssistantState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def route_after_tools(state: AssistantState):
        return NODE_NAMES.get(state["category"], "general_agent")  # back to the same agent

    # ---- assembly -----------------------------------------------------------
    builder = StateGraph(AssistantState)
    builder.add_node("classifier", classify)
    builder.add_node("fallback", fallback)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    for c in CATEGORIES:
        builder.add_node(NODE_NAMES[c], make_agent(c))

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges(
        "classifier", route_after_classifier,
        {**{n: n for n in NODE_NAMES.values()}, "fallback": "fallback"},
    )
    for c in CATEGORIES:
        builder.add_conditional_edges(NODE_NAMES[c], route_after_agent, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", route_after_tools, {n: n for n in NODE_NAMES.values()})
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
# 4. Helpers and CLI
# --------------------------------------------------------------------------- #

DEMO_QUERIES = [
    "What topics are usually covered in a machine learning course?",  # COURSES
    "Write a Python function that computes the factorial of n.",      # PROGRAMMING
    "What is 125 * 38 + 17?",                                         # MATHEMATICS (calculator)
    "How do I request an enrollment certificate?",                    # ADMINISTRATION
    "What is today's date?",                                          # GENERAL (date/time tool)
    "   ",                                                            # error: empty question
]


def ask(assistant, question: str, thread_id: str = "demo"):
    config = {"configurable": {"thread_id": thread_id}}
    result = assistant.invoke({"messages": [HumanMessage(content=question)]}, config)
    print("=" * 70)
    print("USER     :", question)
    print("CATEGORY :", result["category"])
    print("ASSISTANT:", result["messages"][-1].content)
    return result


def main(argv: list[str]) -> int:
    if "--mermaid" in argv:
        os.environ.setdefault("OPENROUTER_API_KEY", "not-needed-to-draw-the-graph")
        print(build_assistant().get_graph().draw_mermaid())
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = getpass("Enter your OpenRouter API key: ")
    assistant = build_assistant()

    if "--demo" in argv:
        for q in DEMO_QUERIES:
            ask(assistant, q)
        ask(assistant, "My name is Ahmed.", thread_id="memory-test")
        ask(assistant, "What is my name?", thread_id="memory-test")
        return 0

    print("University assistant — type 'quit' to exit.")
    while True:
        try:
            q = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in {"quit", "exit"}:
            break
        ask(assistant, q, thread_id="cli")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
