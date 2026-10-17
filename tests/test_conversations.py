"""Example conversations exercising every path of the agent.

Run:  .venv\\Scripts\\python.exe tests\\test_conversations.py
Requires OPENAI_API_KEY in .env (copy .env.example).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.types import Command

from lead_agent.graph import build_graph, make_checkpointer


def send(agent, phone: str, message: str, config: dict) -> str:
    """Send one message; if the graph pauses on interrupt(), simulate the human approving."""
    result = agent.invoke(
        {"phone": phone, "user_message": message},
        config,
    )
    if result.get("__interrupt__"):
        proposal = result["__interrupt__"][0].value
        print(f"   [HITL PAUSE] {proposal['message']}")
        print("   [HUMAN] approving the booking...")
        result = agent.invoke(Command(resume={"approved": True}), config)
    return result.get("agent_reply", "(no reply)")


def main() -> None:
    agent = build_graph(checkpointer=make_checkpointer())

    print("=" * 70)
    print("CONVERSATION 1: Property search with progressive qualification")
    print("=" * 70)
    cfg1 = {"configurable": {"thread_id": "+254712345678"}}
    for msg in [
        "Hi, I'm looking for a place.",
        "I want to rent, budget around 90000 per month",
        "Kilimani, 2 bedrooms",
    ]:
        reply = send(agent, "+254712345678", msg, cfg1)
        print(f"[USER]  {msg}")
        print(f"[AGENT] {reply}\n")

    print("=" * 70)
    print("CONVERSATION 2: Viewing request -> interrupt() -> human approval")
    print("=" * 70)
    cfg2 = {"configurable": {"thread_id": "+254721234567"}}
    for msg in [
        "I want to buy a 3-bedroom apartment in Westlands, Nairobi, budget 28 million",
        "Can I view the first one?",
    ]:
        reply = send(agent, "+254721234567", msg, cfg2)
        print(f"[USER]  {msg}")
        print(f"[AGENT] {reply}\n")

    print("=" * 70)
    print("CONVERSATION 3: General FAQ")
    print("=" * 70)
    cfg3 = {"configurable": {"thread_id": "+254733444555"}}
    reply = send(agent, "+254733444555", "What documents do I need to rent a house?", cfg3)
    print(f"[USER]  What documents do I need to rent a house?")
    print(f"[AGENT] {reply}\n")

    print("=" * 70)
    print("CONVERSATION 4: Human handoff")
    print("=" * 70)
    cfg4 = {"configurable": {"thread_id": "+254755666777"}}
    reply = send(agent, "+254755666777", "This is useless, I want to talk to a real person now!", cfg4)
    print(f"[USER]  This is useless, I want to talk to a real person now!")
    print(f"[AGENT] {reply}\n")

    print("=" * 70)
    print("CONVERSATION 5: Returning lead — same phone continues previous state")
    print("=" * 70)
    # Reuses thread +254712345678: profile (rent / 90k / Kilimani / 2 bedrooms) is already stored,
    # so a vague follow-up needs no re-qualification.
    reply = send(agent, "+254712345678", "Thanks, show me those options again please", cfg1)
    print(f"[USER]  Thanks, show me those options again please  (same phone as Conv 1)")
    print(f"[AGENT] {reply}\n")

    print("All conversations completed.")


if __name__ == "__main__":
    main()
