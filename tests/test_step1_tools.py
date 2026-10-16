"""Step 1 verification: listings JSON loads and search_listings tool filters correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lead_agent.tools import load_listings, search_listings, search_listings_direct


def main() -> None:
    listings = load_listings()
    assert len(listings) == 15, f"Expected 15 listings, got {len(listings)}"
    print(f"OK: loaded {len(listings)} listings")

    mumbai_rent_2bhk = search_listings_direct(location="Mumbai", bedrooms=2, buy_or_rent="rent")
    assert len(mumbai_rent_2bhk) == 1 and mumbai_rent_2bhk[0]["property_id"] == "L002"
    print(f"OK: Mumbai 2BHK rent -> {[l['property_id'] for l in mumbai_rent_2bhk]}")

    bengaluru_buy = search_listings_direct(location="Bengaluru", buy_or_rent="buy")
    assert {l["property_id"] for l in bengaluru_buy} == {"L003", "L008", "L014"}
    print(f"OK: Bengaluru buy -> {[l['property_id'] for l in bengaluru_buy]}")

    budget_match = search_listings_direct(budget=30000000, buy_or_rent="buy")
    assert {l["property_id"] for l in budget_match} == {"L003", "L007"}
    print(f"OK: buy within ~3Cr (+/-15%) -> {[l['property_id'] for l in budget_match]}")

    no_match = search_listings.invoke({"location": "Antarctica"})
    assert "No listings matched" in no_match
    print("OK: no-match message returned")

    tool_result = search_listings.invoke({"bedrooms": 3, "buy_or_rent": "rent"})
    assert "L006" in tool_result and "L015" in tool_result
    print("OK: @tool invoke returns JSON string for 3BHK rentals")

    print("\nStep 1 passed.")


if __name__ == "__main__":
    main()
