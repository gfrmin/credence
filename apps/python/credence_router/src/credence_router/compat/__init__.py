# Role: body
"""LangGraph drop-in compatibility layer.

Swap your import::

    from langgraph.prebuilt import create_react_agent

with::

    from credence_router.compat import create_react_agent

Everything else passes through to real LangGraph.

Requires: pip install credence-router[compat]
"""

from credence_router.compat.prebuilt import create_react_agent  # noqa: F401

__all__ = [
    "END",
    "MemorySaver",
    "MessagesState",
    "StateGraph",
    "ToolNode",
    "create_react_agent",
]


def __getattr__(name: str):
    """Lazy re-export of LangGraph components — only imported when accessed."""
    _langgraph_re_exports = {
        "END": ("langgraph.graph", "END"),
        "StateGraph": ("langgraph.graph", "StateGraph"),
        "MessagesState": ("langgraph.graph.message", "MessagesState"),
        "ToolNode": ("langgraph.prebuilt", "ToolNode"),
        "MemorySaver": ("langgraph.checkpoint.memory", "MemorySaver"),
    }
    if name in _langgraph_re_exports:
        module_path, attr = _langgraph_re_exports[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
