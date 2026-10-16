"""Human handoff node: tells the user a human is taking over and flags the lead."""

from __future__ import annotations

from lead_agent.config import mask_phone
from lead_agent.state import AgentState

HANDOFF_MESSAGE = (
    "I'm handing you over to one of our human property advisors now. "
    "They'll pick up this conversation shortly and help you personally. "
    "Thanks for your patience!"
)


def handoff_node(state: AgentState) -> dict:
    phone = state.get("phone", "")
    print(f"[handoff] lead {mask_phone(phone)} flagged for human agent")  # phone masked in logs
    return {"agent_reply": HANDOFF_MESSAGE, "handed_off": True}
