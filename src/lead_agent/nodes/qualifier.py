"""Qualifier node: collects budget, location, bedrooms, buy_or_rent on the property_search path.

Each turn it extracts any details from the latest user message into the stored profile.
If fields are still missing it asks for the next one and ends the turn (the next user
message re-enters here via the router); once all four fields exist it hands off to search.
The "keep asking" behaviour therefore spans conversation turns, not an in-turn loop.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import END

from lead_agent.config import LLM_MODEL, REQUIRED_LEAD_FIELDS
from lead_agent.prompts import QUALIFIER_SYSTEM_PROMPT
from lead_agent.state import AgentState, LeadProfile

_qualifier_llm: ChatOpenAI | None = None

FIELD_QUESTIONS = {
    "budget": "What is your budget? (total price if buying, monthly rent if renting)",
    "location": "Which city or neighbourhood are you looking in?",
    "bedrooms": "How many bedrooms do you need? (studio, 1, 2, 3, 4+)",
    "buy_or_rent": "Are you looking to buy or rent?",
}

FIELD_FALLBACKS = {
    "budget": None,
    "location": None,
    "bedrooms": None,
    "buy_or_rent": None,
}


def get_qualifier_llm() -> ChatOpenAI:
    global _qualifier_llm
    if _qualifier_llm is None:
        _qualifier_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return _qualifier_llm


def extract_profile(message: str, previous: LeadProfile) -> LeadProfile:
    """Ask the LLM to update the lead profile with details from the latest message."""
    llm = get_qualifier_llm().with_structured_output(LeadProfile)
    prompt = (
        f"{QUALIFIER_SYSTEM_PROMPT}\n\n"
        f"Previously collected profile: {previous.model_dump()}\n"
        f"Latest user message: {message}"
    )
    extracted = llm.invoke([{"role": "user", "content": prompt}])
    merged = previous.model_dump()
    for field in REQUIRED_LEAD_FIELDS:
        new_value = getattr(extracted, field)
        if new_value is not None:
            merged[field] = new_value
    return LeadProfile(**merged)


def qualifier_node(state: AgentState) -> dict:
    previous = state.get("lead") or LeadProfile()
    profile = extract_profile(state["user_message"], previous)

    if profile.missing_fields:
        next_field = profile.missing_fields[0]
        reply = f"To find the right property, I need a bit more info: {FIELD_QUESTIONS[next_field]}"
        return {"lead": profile, "agent_reply": reply}

    reply = (
        "Great, I have everything I need: "
        f"{'rent' if profile.buy_or_rent == 'rent' else 'buy'}, "
        f"{profile.bedrooms} bedroom(s) in {profile.location} "
        f"around INR {profile.budget:,}. Let me search my listings."
    )
    return {"lead": profile, "agent_reply": reply}


def qualification_gate(state: AgentState) -> str:
    """Conditional edge: 'search' when the profile is complete, otherwise end the turn and wait
    for the user to answer the qualifying question (the next message returns here via the router)."""
    profile = state.get("lead") or LeadProfile()
    return "search" if not profile.missing_fields else END
