from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


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
    model_config = ConfigDict(extra="forbid")

    type: ClauseType
    span: str = Field(min_length=1)


class ClauseList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clauses: list[Clause]


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)
    clause_types: list[ClauseType] | None = None


class ExtractResponse(BaseModel):
    clauses: list[Clause]
    latency_ms: float
    model_version: str
