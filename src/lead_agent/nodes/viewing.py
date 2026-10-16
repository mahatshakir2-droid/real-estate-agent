"""Viewing node: proposes a slot, pauses with interrupt() for human approval,
and only confirms after the human resumes the graph with an approve/reject decision."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from lead_agent.config import LLM_MODEL
from lead_agent.prompts import VIEWING_SYSTEM_PROMPT
from lead_agent.state import AgentState, BookingProposal


def propose_time(now: datetime | None = None) -> str:
    """Deterministic slot proposal: tomorrow at 11:00 local time (simple, no calendar integration)."""
    now = now or datetime.now()
    slot = now + timedelta(days=1)
    return f"{slot.strftime('%A, %d %B %Y')} at 11:00 AM"


def resolve_property_id(state: AgentState, message: str) -> str | None:
    """Property to view: explicit L0xx id in the message, else first match from the last search."""
    match = re.search(r"\bL0\d{2}\b", message, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()
    matched = state.get("matched_property_ids") or []
    return matched[0] if matched else None


def _decision_message(proposal: BookingProposal, approved: bool) -> str:
    decision = "approved" if approved else "rejected"
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        prompt = VIEWING_SYSTEM_PROMPT.format(
            decision=decision, property_id=proposal.property_id, proposed_time=proposal.proposed_time
        )
        reply = llm.invoke([SystemMessage(content=prompt)])
        return reply.text
    except Exception:
        if approved:
            return (
                f"Your viewing of {proposal.property_id} on {proposal.proposed_time} is confirmed! "
                "Our agent will meet you at the property. Please carry a photo ID."
            )
        return (
            f"Sorry, the slot {proposal.proposed_time} for {proposal.property_id} is no longer available. "
            "Would you like me to propose another time?"
        )


def viewing_node(state: AgentState) -> dict:
    proposal: BookingProposal | None = state.get("last_proposal")

    # First pass: no proposal yet, or a pending one that was just resumed.
    if proposal is None or proposal.status == "pending":
        if proposal is None:
            property_id = resolve_property_id(state, state["user_message"])
            if property_id is None:
                return {
                    "agent_reply": (
                        "I'd be happy to arrange a viewing! Which property are you interested in? "
                        "You can share its property_id (e.g. L003)."
                    )
                }
            proposal = BookingProposal(
                property_id=property_id, proposed_time=propose_time(), status="pending"
            )

        # Pauses the graph. Writes made before this call (none yet) are persisted;
        # on resume, interrupt() returns the human's decision payload.
        decision = interrupt(
            {
                "type": "viewing_approval",
                "property_id": proposal.property_id,
                "proposed_time": proposal.proposed_time,
                "message": (
                    f"A viewing slot has been proposed for {proposal.property_id} "
                    f"on {proposal.proposed_time}. Approve or reject?"
                ),
            }
        )

        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        proposal.status = "approved" if approved else "rejected"
        return {
            "last_proposal": proposal,
            "agent_reply": _decision_message(proposal, approved),
        }

    # Safety net: a proposal already resolved before this node ran (shouldn't normally happen).
    return {"last_proposal": proposal, "agent_reply": _decision_message(proposal, proposal.status == "approved")}
