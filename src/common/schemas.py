from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    GOVERNING_LAW = "Governing Law"
    INDEMNIFICATION = "Indemnification"
    NON_COMPETE = "Non-Compete"
    TERMINATION_FOR_CONVENIENCE = "Termination for Convenience"
    LIABILITY_CAP = "Liability Cap"
    EXCLUSIVITY = "Exclusivity"
    IP_ASSIGNMENT = "IP Assignment"
    CONFIDENTIALITY = "Confidentiality"
    CHANGE_OF_CONTROL = "Change of Control"
    AUTO_RENEWAL = "Auto-Renewal"


class Clause(BaseModel):
    type: ClauseType
    span: str = Field(min_length=1)


class ClauseList(BaseModel):
    clauses: list[Clause] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)
    clause_types: list[ClauseType] | None = None


class ExtractResponse(BaseModel):
    clauses: list[Clause]
    latency_ms: float
    model_version: str
