"""Key-free structural tests: stub the LLM calls so we can verify graph wiring,
the qualifier loop, tool-calling search, checkpoint memory, and interrupt/resume HITL.

Run:  .venv\\Scripts\\python.exe tests\\test_graph_structure.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage
from langgraph.types import Command

from lead_agent.graph import build_graph, make_checkpointer
from lead_agent.nodes import faq as faq_mod
from lead_agent.nodes import qualifier as qualifier_mod
from lead_agent.nodes import router as router_mod
from lead_agent.nodes import search as search_mod
from lead_agent.nodes import viewing as viewing_mod
from lead_agent.state import LeadProfile

INTENT_SCRIPT: list[str] = []


def fake_classify(message, history=None):
    return router_mod.RouteDecision(intent=INTENT_SCRIPT.pop(0), reasoning="stub")


def fake_extract(message, previous):
    return previous  # pre-seed the profile in tests; extraction itself is covered by live tests


class FakeSearchLLM:
    """First call returns a tool call, second returns a text reply."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls % 2 == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_listings",
                        "args": {"location": "Nairobi", "bedrooms": 2, "buy_or_rent": "rent"},
                        "id": f"call_{self.calls}",
                    }
                ],
            )
        return AIMessage(content="Found L002: fully furnished 2-bedroom apartment in Kilimani at KSh 85,000/month.")


def patch_all():
    router_mod.classify_intent = fake_classify
    qualifier_mod.extract_profile = fake_extract
    search_mod.get_search_llm = lambda: FakeSearchLLM()
    faq_mod.answer_faq = lambda q: "FAQ-STUB-ANSWER"
    # Avoid a live LLM call in the viewing node; keep the deterministic fallback wording.
    viewing_mod._decision_message = lambda proposal, approved: (
        f"Viewing of {proposal.property_id} is confirmed for {proposal.proposed_time}."
        if approved
        else f"Sorry, the slot {proposal.proposed_time} for {proposal.property_id} is unavailable."
    )


def test_search_with_memory():
    patch_all()
    INTENT_SCRIPT.clear()
    INTENT_SCRIPT.extend(["property_search"])
    agent = build_graph(make_checkpointer())
    cfg = {"configurable": {"thread_id": "+254700000001"}}

    state = {"phone": "+254700000001", "user_message": "find rentals", "lead": LeadProfile(
        budget=85000, location="Nairobi", bedrooms=2, buy_or_rent="rent")}
    out = agent.invoke(state, cfg)
    assert "L002" in out["agent_reply"], out["agent_reply"]
    assert out["matched_property_ids"] == ["L002"], out["matched_property_ids"]
    print("OK: search path resolves via tool-calling and returns matched ids")

    # New thread resumes the stored state: no lead passed, still finds the same listing.
    INTENT_SCRIPT.append("property_search")
    out2 = agent.invoke({"phone": "+254700000001", "user_message": "again"}, cfg)
    assert out2["matched_property_ids"] == ["L002"], out2["matched_property_ids"]
    print("OK: checkpointer memory persisted the lead across messages")


def test_qualifier_loop():
    patch_all()
    INTENT_SCRIPT.clear()
    INTENT_SCRIPT.extend(["property_search"])

    def partial_extract(message, previous):
        return LeadProfile(location="Nairobi", buy_or_rent="rent")  # budget + bedrooms still missing

    qualifier_mod.extract_profile = partial_extract
    agent = build_graph(make_checkpointer())
    cfg = {"configurable": {"thread_id": "+254700000002"}}
    out = agent.invoke({"phone": "+254700000002", "user_message": "rent something in Nairobi"}, cfg)
    # Profile incomplete -> qualifier loops -> asks for a missing field, never reaches search.
    assert out["intent"] == "property_search"
    assert "budget" in out["agent_reply"], out["agent_reply"]
    assert not out.get("matched_property_ids")
    print("OK: qualifier loops and asks for the next missing field")


def test_viewing_interrupt_approve():
    patch_all()
    INTENT_SCRIPT.clear()
    INTENT_SCRIPT.extend(["viewing_request"])
    agent = build_graph(make_checkpointer())
    cfg = {"configurable": {"thread_id": "+254700000003"}}

    out = agent.invoke(
        {"phone": "+254700000003", "user_message": "I want to view L010",
         "matched_property_ids": ["L010"]},
        cfg,
    )
    interrupts = out.get("__interrupt__")
    assert interrupts, "expected the graph to pause on interrupt()"
    payload = interrupts[0].value
    assert payload["type"] == "viewing_approval"
    assert payload["property_id"] == "L010", payload
    print(f"OK: viewing paused for HITL approval -> {payload['message']}")

    resumed = agent.invoke(Command(resume={"approved": True}), cfg)
    assert resumed["last_proposal"].status == "approved"
    assert "confirmed" in resumed["agent_reply"].lower() or "L010" in resumed["agent_reply"]
    print(f"OK: approved resume confirms booking -> {resumed['agent_reply']}")


def test_viewing_interrupt_reject():
    patch_all()
    INTENT_SCRIPT.clear()
    INTENT_SCRIPT.extend(["viewing_request"])
    agent = build_graph(make_checkpointer())
    cfg = {"configurable": {"thread_id": "+254700000004"}}
    out = agent.invoke({"phone": "+254700000004", "user_message": "view L001"}, cfg)
    assert out.get("__interrupt__")
    resumed = agent.invoke(Command(resume={"approved": False}), cfg)
    assert resumed["last_proposal"].status == "rejected"
    print(f"OK: rejected resume -> {resumed['agent_reply']}")


def test_faq_and_handoff():
    patch_all()
    agent = build_graph(make_checkpointer())

    INTENT_SCRIPT.clear()
    INTENT_SCRIPT.extend(["general_faq"])
    cfg = {"configurable": {"thread_id": "+254700000005"}}
    out = agent.invoke({"phone": "+254700000005", "user_message": "buy or rent?"}, cfg)
    assert out["agent_reply"] == "FAQ-STUB-ANSWER"
    print("OK: faq path returns grounded answer")

    INTENT_SCRIPT.clear()
    INTENT_SCRIPT.extend(["human_handoff"])
    cfg2 = {"configurable": {"thread_id": "+254700000006"}}
    out = agent.invoke({"phone": "+254700000006", "user_message": "get me a human"}, cfg2)
    assert out["handed_off"] is True
    assert "human" in out["agent_reply"].lower()
    print("OK: handoff path flags the lead and notifies the user")


if __name__ == "__main__":
    test_search_with_memory()
    test_qualifier_loop()
    test_viewing_interrupt_approve()
    test_viewing_interrupt_reject()
    test_faq_and_handoff()
    print("\nAll structural tests passed (no API key needed).")
