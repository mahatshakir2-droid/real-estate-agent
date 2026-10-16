"""Router node: classifies each message into one intent using structured output."""

from __future__ import annotations

from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from lead_agent.config import LLM_MODEL
from lead_agent.prompts import ROUTER_SYSTEM_PROMPT
from lead_agent.state import AgentState

Intent = Literal["property_search", "viewing_request", "general_faq", "human_handoff"]


class RouteDecision(BaseModel):
    """Structured classification of the user's message."""

    intent: Intent = Field(description="The single best intent for this message")
    reasoning: str = Field(description="One short sentence explaining the choice")


_router_llm: ChatOpenAI | None = None


def get_router_llm() -> ChatOpenAI:
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return _router_llm


def classify_intent(message: str, history: list[dict] | None = None) -> RouteDecision:
    llm = get_router_llm().with_structured_output(RouteDecision)
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})
    return llm.invoke(messages)


def route_by_intent(state: AgentState) -> str:
    """Conditional edge function: maps state['intent'] to a graph node name."""
    intent = state.get("intent") or "general_faq"  # fallback: never crash on unknown intent
    return intent


def router_node(state: AgentState) -> dict:
    try:
        decision = classify_intent(state["user_message"])
        intent = decision.intent
    except Exception:
        intent = "general_faq"  # LLM/structured-output failure fallback
    return {"intent": intent}
