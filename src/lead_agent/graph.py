"""Graph construction: wires nodes, conditional edges, and the checkpointer.

Topology:
    START -> router -> (intent) -> qualifier | viewing | faq | handoff
    qualifier -> (missing fields?) -> qualifier | search
    search / faq / handoff -> END
    viewing -> interrupt() pause -> resume -> END

Memory: MemorySaver checkpointer keyed by thread_id (= phone number), so a returning
lead with the same phone continues from their previous conversation state.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer

from lead_agent.nodes.faq import faq_node
from lead_agent.nodes.handoff import handoff_node
from lead_agent.nodes.qualifier import qualification_gate, qualifier_node
from lead_agent.nodes.router import route_by_intent, router_node
from lead_agent.nodes.search import search_node
from lead_agent.nodes.viewing import viewing_node
from lead_agent.state import AgentState


def build_graph(checkpointer: Checkpointer | None = None):
    """Build and compile the agent graph. Pass a checkpointer for persistent memory."""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("qualifier", qualifier_node)
    graph.add_node("search", search_node)
    graph.add_node("viewing", viewing_node)
    graph.add_node("faq", faq_node)
    graph.add_node("handoff", handoff_node)

    graph.add_edge(START, "router")

    # Router -> intent-specific node
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "property_search": "qualifier",
            "viewing_request": "viewing",
            "general_faq": "faq",
            "human_handoff": "handoff",
        },
    )

    # Qualifier either moves on to search (profile complete) or ends the turn to ask for
    # the next missing field; the next user message re-enters via the router.
    graph.add_conditional_edges(
        "qualifier",
        qualification_gate,
        {"search": "search", END: END},
    )

    graph.add_edge("search", END)
    graph.add_edge("viewing", END)
    graph.add_edge("faq", END)
    graph.add_edge("handoff", END)

    return graph.compile(checkpointer=checkpointer)


# Pydantic models stored in state must be whitelisted for checkpoint (de)serialization.
STATE_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("lead_agent.state", "LeadProfile"),
        ("lead_agent.state", "BookingProposal"),
    ]
)


def make_checkpointer() -> MemorySaver:
    """In-memory checkpointer (thread_id = phone number). Swap for a DB saver in production."""
    return MemorySaver(serde=STATE_SERDE)
