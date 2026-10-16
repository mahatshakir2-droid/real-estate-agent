"""Search node: LLM tool-calling loop over search_listings, then a friendly reply."""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from lead_agent.config import LLM_MODEL
from lead_agent.prompts import SEARCH_SYSTEM_PROMPT
from lead_agent.state import AgentState
from lead_agent.tools import search_listings

_search_llm: ChatOpenAI | None = None


def get_search_llm() -> ChatOpenAI:
    global _search_llm
    if _search_llm is None:
        _search_llm = ChatOpenAI(model=LLM_MODEL, temperature=0).bind_tools([search_listings])
    return _search_llm


def _extract_property_ids(tool_result: str) -> list[str]:
    import json

    try:
        listings = json.loads(tool_result)
        return [l["property_id"] for l in listings]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def search_node(state: AgentState) -> dict:
    profile = (state.get("lead") or None)
    llm = get_search_llm()

    messages = [
        SystemMessage(content=SEARCH_SYSTEM_PROMPT.format(lead_profile=profile.model_dump() if profile else {})),
        HumanMessage(content=state["user_message"]),
    ]
    ai_msg = llm.invoke(messages)

    matched_ids: list[str] = []
    tool_content = ""
    for tool_call in ai_msg.tool_calls:
        result = search_listings.invoke(tool_call["args"])
        tool_content = result if isinstance(result, str) else str(result)
        matched_ids.extend(_extract_property_ids(tool_content))
        messages.append(ai_msg)
        messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": tool_content})
        ai_msg = llm.invoke(messages)

    reply = ai_msg.text
    if not matched_ids and tool_content:
        reply = f"{tool_content}\n\n{reply}"

    return {"agent_reply": reply, "matched_property_ids": matched_ids}
