"""Tools available to the agent. Currently: property listing search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LISTINGS_PATH = DATA_DIR / "listings.json"


def load_listings() -> list[dict[str, Any]]:
    with LISTINGS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _match_listing(
    listing: dict[str, Any],
    budget: int | None,
    location: str | None,
    bedrooms: int | None,
    buy_or_rent: str | None,
    tolerance: float = 0.15,
) -> bool:
    if buy_or_rent and listing["buy_or_rent"].lower() != buy_or_rent.lower():
        return False
    if bedrooms is not None and listing["bedrooms"] != bedrooms:
        return False
    if location and location.strip().lower() not in listing["location"].lower():
        return False
    if budget is not None:
        upper = budget * (1 + tolerance)
        lower = budget * (1 - tolerance)
        if not (lower <= listing["price"] <= upper):
            return False
    return True


@tool
def search_listings(
    budget: int | None = None,
    location: str | None = None,
    bedrooms: int | None = None,
    buy_or_rent: str | None = None,
) -> str:
    """Search property listings by budget (total price for buy, monthly rent for rent),
    location (city or neighbourhood substring, e.g. 'Kilimani' or 'Nairobi'),
    bedrooms (exact count), and buy_or_rent ('buy' or 'rent').

    All provided filters are applied together. Budget matches use a +/-15% tolerance.
    Returns a JSON-style list of matching listings with a short note when nothing matches.
    """
    listings = load_listings()
    matches = [
        l
        for l in listings
        if _match_listing(l, budget=budget, location=location, bedrooms=bedrooms, buy_or_rent=buy_or_rent)
    ]
    if not matches:
        return (
            "No listings matched all filters. Consider relaxing budget, location or bedroom criteria. "
            f"Received filters: budget={budget}, location={location}, bedrooms={bedrooms}, buy_or_rent={buy_or_rent}."
        )
    matches.sort(key=lambda l: l["price"])
    return json.dumps(matches, ensure_ascii=False, indent=2)


def search_listings_direct(
    budget: int | None = None,
    location: str | None = None,
    bedrooms: int | None = None,
    buy_or_rent: str | None = None,
) -> list[dict[str, Any]]:
    """Non-tool wrapper used by tests: returns matching listings as Python dicts."""
    listings = load_listings()
    return [
        l
        for l in listings
        if _match_listing(l, budget=budget, location=location, bedrooms=bedrooms, buy_or_rent=buy_or_rent)
    ]
