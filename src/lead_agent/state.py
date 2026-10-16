"""LangGraph state schema.

AgentState is a TypedDict (LangGraph's preferred state shape: every node returns a
partial update dict, and keys with None values are ignored by default reducer behaviour
for TypedDict states). LeadProfile and BookingProposal are Pydantic models embedded
in the state for validation and structured storage.
"""

from __future__ import annotations

from typing import Literal, NotRequired, Optional, TypedDict

from pydantic import BaseModel, Field


class LeadProfile(BaseModel):
    """Collected qualification details for a lead. Fields start unset."""

    budget: Optional[int] = Field(default=None, description="Total price for buy, monthly rent for rent, in INR")
    location: Optional[str] = Field(default=None, description="Preferred city or neighbourhood")
    bedrooms: Optional[int] = Field(default=None, description="Number of bedrooms")
    buy_or_rent: Optional[Literal["buy", "rent"]] = Field(default=None, description="Intent: buy or rent")

    @property
    def missing_fields(self) -> list[str]:
        return [name for name, value in self.model_dump().items() if value is None]


class BookingProposal(BaseModel):
    """A proposed viewing slot awaiting human approval."""

    property_id: str
    proposed_time: str
    status: Literal["pending", "approved", "rejected"] = "pending"


class AgentState(TypedDict):
    phone: str
    user_message: str
    intent: NotRequired[Optional[str]]
    lead: NotRequired[LeadProfile]
    matched_property_ids: NotRequired[list[str]]
    agent_reply: NotRequired[str]
    last_proposal: NotRequired[Optional[BookingProposal]]
    handed_off: NotRequired[bool]
