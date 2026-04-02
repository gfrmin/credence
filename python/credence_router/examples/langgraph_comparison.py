#!/usr/bin/env python3
"""Side-by-side comparison: LangGraph ReAct vs credence-router with Ollama.

Runs the same questions through both routing approaches using a local llama3.1
model via Ollama, then prints a comparison table showing LLM calls saved and
latency differences.

Requirements:
    uv sync --extra demo
    ollama pull llama3.1

Usage:
    uv run python examples/langgraph_comparison.py
"""

from __future__ import annotations

import time
import warnings
from typing import Any

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# -- Tools -------------------------------------------------------------------

@tool
def calculator(expr: str) -> str:
    """Evaluate a mathematical expression. Use for arithmetic, percentages, etc."""
    allowed = set("0123456789+-*/.() %")
    sanitised = expr.replace("^", "**").replace("%", "/100*")
    if not all(c in allowed or c == '*' for c in sanitised):
        return f"Cannot evaluate: {expr}"
    try:
        return str(round(eval(sanitised), 6))  # noqa: S307 — sandboxed to digits+operators
    except Exception:
        return f"Cannot evaluate: {expr}"


CAPITALS = {
    "france": "Paris",
    "germany": "Berlin",
    "japan": "Tokyo",
    "brazil": "Brasília",
    "australia": "Canberra",
    "canada": "Ottawa",
    "italy": "Rome",
    "spain": "Madrid",
    "south korea": "Seoul",
    "egypt": "Cairo",
}


@tool
def lookup_capital(country: str) -> str:
    """Look up the capital city of a country."""
    result = CAPITALS.get(country.lower().strip())
    if result:
        return result
    return f"Capital not found for: {country}"


POPULATIONS = {
    "paris": "2.1 million",
    "berlin": "3.6 million",
    "tokyo": "14 million",
    "brasília": "3.0 million",
    "canberra": "460,000",
    "ottawa": "1.0 million",
    "rome": "2.8 million",
    "madrid": "3.3 million",
    "seoul": "9.7 million",
    "cairo": "10.1 million",
}


@tool
def lookup_population(city: str) -> str:
    """Look up the population of a city."""
    result = POPULATIONS.get(city.lower().strip())
    if result:
        return result
    return f"Population not found for: {city}"


# -- LLM call counter -------------------------------------------------------

class CountedChatOllama(ChatOllama):
    """ChatOllama subclass that counts invoke() and bind_tools().invoke() calls."""

    _call_counter: dict[str, int] = {}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._call_counter = {"calls": 0}

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        self._call_counter["calls"] += 1
        return super().invoke(*args, **kwargs)

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        bound = super().bind_tools(*args, **kwargs)
        counter = self._call_counter
        original_invoke = bound.invoke

        def counting_bound_invoke(*a: Any, **kw: Any) -> Any:
            counter["calls"] += 1
            return original_invoke(*a, **kw)

        # RunnableBinding is Pydantic — bypass __setattr__ validation
        object.__setattr__(bound, "invoke", counting_bound_invoke)
        return bound

    @property
    def counter(self) -> dict[str, int]:
        return self._call_counter


# -- Questions ---------------------------------------------------------------

QUESTIONS = [
    "What is 15% of 240?",
    "What is the capital of France?",
    "What is the population of Tokyo?",
    "What is 127 * 43?",
    "What is the capital of Brazil?",
    "What is 2^10?",
    "What is the population of Seoul?",
    "What is the capital of Egypt?",
]


# -- Run comparison ----------------------------------------------------------

def run_agent(graph: Any, question: str, model: CountedChatOllama) -> dict[str, Any]:
    """Run a single question through a graph, tracking calls and time."""
    counter = model.counter
    counter["calls"] = 0
    t0 = time.monotonic()
    result = graph.invoke({"messages": [("human", question)]})
    elapsed = time.monotonic() - t0
    final = result["messages"][-1].content
    # Find which tools were called
    tools_used = [
        m.name for m in result["messages"]
        if hasattr(m, "type") and m.type == "tool"
    ]
    return {
        "answer": final[:80] if isinstance(final, str) else str(final)[:80],
        "tools": tools_used,
        "llm_calls": counter["calls"],
        "time": elapsed,
    }


def main() -> None:
    tools = [calculator, lookup_capital, lookup_population]

    # LangGraph (LLM routes)
    from langgraph.prebuilt import create_react_agent as lg_create

    lg_model = CountedChatOllama(model="llama3.1")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        lg_graph = lg_create(lg_model, tools)

    # credence-router (VOI routes)
    from credence_router.compat import create_react_agent as cr_create

    cr_model = CountedChatOllama(model="llama3.1")
    cr_graph = cr_create(cr_model, tools)

    # Run all questions
    lg_results: list[dict[str, Any]] = []
    cr_results: list[dict[str, Any]] = []

    for q in QUESTIONS:
        print(f"  Running: {q}")
        lg_results.append(run_agent(lg_graph, q, lg_model))
        cr_results.append(run_agent(cr_graph, q, cr_model))

    # Print comparison table
    print()
    print("=" * 100)
    print(f"{'Question':<30} | {'LangGraph':<32} | {'credence-router':<32}")
    print("-" * 100)

    total_lg_calls = 0
    total_cr_calls = 0
    total_lg_time = 0.0
    total_cr_time = 0.0

    for q, lg, cr in zip(QUESTIONS, lg_results, cr_results):
        lg_tools = ", ".join(lg["tools"]) or "none"
        cr_tools = ", ".join(cr["tools"]) or "none"
        lg_summary = f"{lg_tools}, {lg['llm_calls']} LLM, {lg['time']:.1f}s"
        cr_summary = f"{cr_tools}, {cr['llm_calls']} LLM, {cr['time']:.1f}s"
        print(f"{q:<30} | {lg_summary:<32} | {cr_summary:<32}")
        total_lg_calls += lg["llm_calls"]
        total_cr_calls += cr["llm_calls"]
        total_lg_time += lg["time"]
        total_cr_time += cr["time"]

    print("-" * 100)
    print(
        f"{'TOTAL':<30} | "
        f"{total_lg_calls} LLM calls, {total_lg_time:.1f}s{'':<9} | "
        f"{total_cr_calls} LLM calls, {total_cr_time:.1f}s"
    )
    print("=" * 100)

    saved = total_lg_calls - total_cr_calls
    if saved > 0:
        print(f"\ncredence-router saved {saved} LLM calls ({saved/total_lg_calls:.0%} fewer)")
    speedup = total_lg_time / total_cr_time if total_cr_time > 0 else 0
    if speedup > 1:
        print(f"Total speedup: {speedup:.1f}x faster")

    print("\n--- The only change: ---")
    print('- from langgraph.prebuilt import create_react_agent')
    print('+ from credence_router.compat import create_react_agent')


if __name__ == "__main__":
    main()
