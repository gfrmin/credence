# Role: body
"""Drop-in replacement for langgraph.prebuilt.create_react_agent.

Builds a real LangGraph StateGraph but swaps the LLM routing node with
credence-router's VOI-based selection. Everything else — ToolNode, edges,
streaming, checkpointing — is pure LangGraph.
"""

from __future__ import annotations

from credence_router.compat.routing_node import make_credence_routing_node
from credence_router.compat.tool_adapter import ToolRoutingAdapter
from credence_router.router import Router


def create_react_agent(model, tools, *, checkpointer=None, **kwargs):
    """Drop-in replacement for langgraph.prebuilt.create_react_agent.

    Builds a real LangGraph StateGraph but replaces the LLM routing node
    with credence-router's VOI-based selection. Everything else — ToolNode,
    edges, streaming, checkpointing — is pure LangGraph.

    Cost savings: eliminates $0.001-0.004 per routing decision.

    Args:
        model: A LangChain chat model (e.g. ChatAnthropic). Used for argument
            extraction and answer synthesis, NOT for routing.
        tools: List of LangChain tools (@tool decorated functions or BaseTool).
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver).
        **kwargs: Passed through to LangGraph's workflow.compile().

    Returns:
        A compiled LangGraph graph with invoke(), stream(), ainvoke(), astream().
    """
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import MessagesState
    from langgraph.prebuilt import ToolNode

    # Build credence-router from tool adapters
    adapters = [ToolRoutingAdapter(t, i, len(tools)) for i, t in enumerate(tools)]
    router = Router(tools=adapters)

    # Build routing node and standard tool node
    routing_node = make_credence_routing_node(model, tools, router)
    tool_node = ToolNode(tools)

    def should_continue(state: dict) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # Assemble standard LangGraph graph with our routing node
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", routing_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    graph = workflow.compile(checkpointer=checkpointer, **kwargs)
    graph._credence_router = router  # Expose for feedback and state inspection
    return graph
