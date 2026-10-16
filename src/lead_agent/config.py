"""Central configuration: model selection, constants, and log-safe helpers."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

REQUIRED_LEAD_FIELDS = ("budget", "location", "bedrooms", "buy_or_rent")

FAQ_PATH = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "faqs.json"


def mask_phone(phone: str) -> str:
    """Return a log-safe version of a phone number, e.g. +15551237890 -> +1555***7890."""
    digits = phone.strip()
    if len(digits) <= 4:
        return "***"
    return f"{digits[:5]}***{digits[-4:]}"
