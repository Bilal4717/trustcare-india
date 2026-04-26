from typing import List, Optional, Literal, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class HealthcareState(TypedDict):
    messages: List[BaseMessage]
    intent: Optional[Literal["query", "audit", "desert", "trust", "out_of_scope"]]
    retrieved_docs: Optional[List[dict]]
    answer: Optional[str]
    audit_result: Optional[dict]
    trust_score: Optional[float]
    trust_flags: Optional[List[str]]
    desert_regions: Optional[List[dict]]
    validated: Optional[bool]
    correction_notes: Optional[str]
    chain_of_thought: Optional[List[str]]
    # Row-level citations (verbatim snippets from retrieved facility text)
    source_citations: Optional[List[dict]]
    # Unified confidence and explainability metadata
    confidence_score: Optional[float]
    confidence_band: Optional[str]
    reason_codes: Optional[List[str]]
    trust_breakdown: Optional[List[dict]]
    trust_evidence_map: Optional[List[dict]]
    planner_summary: Optional[List[dict]]
    validation_attempts: Optional[int]
    correction_applied: Optional[bool]


class FacilityCapabilities(BaseModel):
    has_icu: bool = Field(default=False)
    has_emergency: bool = Field(default=False)
    has_surgery: bool = Field(default=False)
    has_dialysis: bool = Field(default=False)
    has_oncology: bool = Field(default=False)
    has_neonatal: bool = Field(default=False)
    num_doctors: Optional[int] = Field(default=None)
    bed_capacity: Optional[int] = Field(default=None)
    operates_24_7: bool = Field(default=False)
    confidence_note: str = Field(default="")


class TrustAssessment(BaseModel):
    score: float = Field(description="0.0 (untrustworthy) to 1.0 (fully consistent)")
    flags: List[str] = Field(default_factory=list)
    explanation: str = Field(default="")

