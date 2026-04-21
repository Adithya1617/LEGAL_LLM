import pytest
from pydantic import ValidationError

from src.common.schemas import (
    Clause,
    ClauseList,
    ClauseType,
    ExtractRequest,
    ExtractResponse,
)


def test_clause_type_has_exactly_ten_members():
    assert len(list(ClauseType)) == 10


def test_clause_type_contains_expected_values():
    values = {c.value for c in ClauseType}
    assert "Governing Law" in values
    assert "Indemnification" in values
    assert "Auto-Renewal" in values


def test_clause_accepts_valid_type_and_span():
    c = Clause(type=ClauseType.GOVERNING_LAW, span="This Agreement is governed by Delaware law.")
    assert c.span.startswith("This Agreement")


def test_clause_rejects_empty_span():
    with pytest.raises(ValidationError):
        Clause(type=ClauseType.GOVERNING_LAW, span="")


def test_clause_list_allows_empty():
    cl = ClauseList(clauses=[])
    assert cl.clauses == []


def test_clause_list_round_trips_json():
    payload = '{"clauses":[{"type":"Governing Law","span":"x"}]}'
    cl = ClauseList.model_validate_json(payload)
    assert cl.clauses[0].type == ClauseType.GOVERNING_LAW
    assert cl.model_dump_json() == payload


def test_extract_request_defaults_clause_types_to_none():
    req = ExtractRequest(text="hello")
    assert req.clause_types is None


def test_extract_response_fields():
    resp = ExtractResponse(
        clauses=[],
        latency_ms=12.5,
        model_version="llama-3.2-3b-legal@abc1234",
    )
    assert resp.latency_ms == 12.5
