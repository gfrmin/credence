"""Custom LangGraph node that uses credence-router for tool selection.

Replaces the LLM's tool-selection step with credence-router's VOI calculation.
The rest of the LangGraph graph (ToolNode, edges, streaming) works unchanged.
"""

from __future__ import annotations

from credence_router.router import Router


def _extract_question(messages: list) -> str:
    """Extract the most recent human message from the conversation."""
    for msg in reversed(messages):
        if isinstance(msg, tuple):
            if msg[0] == "human":
                return msg[1]
        elif hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""


def make_credence_routing_node(model, tools: list, router: Router):
    """Build a LangGraph node that uses credence-router for tool selection.

    In standard LangGraph ReAct, the "model" node calls the LLM which returns
    an AIMessage with tool_calls. The LLM decides which tool(s) to call.

    Our replacement:
    1. Router.solve() picks which tool to call (free, <1ms)
    2. Model formats the tool arguments (still needs LLM for arg extraction)
    3. Returns AIMessage with tool_calls, same as LangGraph expects

    After tool execution, the model synthesises the final answer from the
    tool result — same as standard ReAct.
    """

    def routing_node(state: dict) -> dict:
        messages = state["messages"]
        last_msg = messages[-1]

        # After tool execution: let model synthesise the answer
        if hasattr(last_msg, "type") and last_msg.type == "tool":
            response = model.invoke(messages)
            return {"messages": [response]}

        # Route: select best tool via VOI (<1ms, $0)
        question = _extract_question(messages)
        candidates = tuple(t.name for t in tools)
        answer = router.solve(question, candidates)

        if answer.choice is not None:
            selected_tool = tools[answer.choice]
            # Bind only the selected tool — cheaper than evaluating all schemas
            response = model.bind_tools([selected_tool]).invoke(messages)
        else:
            # No tool needed — model responds directly
            response = model.invoke(messages)

        return {"messages": [response]}

    return routing_node
