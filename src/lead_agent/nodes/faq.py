"""FAQ node: answers general questions with an LLM grounded strictly on data/faqs.json."""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from lead_agent.config import FAQ_PATH, LLM_MODEL
from lead_agent.prompts import FAQ_SYSTEM_PROMPT
from lead_agent.state import AgentState

_faq_llm: ChatOpenAI | None = None


def load_faqs() -> list[dict]:
    with FAQ_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def get_faq_llm() -> ChatOpenAI:
    global _faq_llm
    if _faq_llm is None:
        _faq_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return _faq_llm


def answer_faq(question: str) -> str:
    faqs = load_faqs()
    faq_kb = "\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in faqs)
    llm = get_faq_llm()
    messages = [
        SystemMessage(content=FAQ_SYSTEM_PROMPT.format(faq_kb=faq_kb)),
        HumanMessage(content=question),
    ]
    reply = llm.invoke(messages)
    return reply.text


def faq_node(state: AgentState) -> dict:
    try:
        reply = answer_faq(state["user_message"])
    except Exception:
        reply = (
            "I'm having trouble fetching that answer right now. "
            "Would you like me to connect you with a human agent?"
        )
    return {"agent_reply": reply}
