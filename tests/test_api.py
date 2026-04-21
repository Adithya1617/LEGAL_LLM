from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.common.schemas import Clause, ClauseType, ExtractResponse
from src.serve import api as api_module


@pytest.fixture
def client(monkeypatch):
    mock_extractor = MagicMock()
    mock_extractor.model_version = "mock-v1"
    mock_extractor.extract.return_value = ExtractResponse(
        clauses=[Clause(type=ClauseType.GOVERNING_LAW, span="Delaware law.")],
        latency_ms=5.0,
        model_version="mock-v1",
    )

    def fake_build_extractor(*args, **kwargs):
        return mock_extractor

    monkeypatch.setattr(api_module, "build_extractor", fake_build_extractor)
    with TestClient(api_module.app) as c:
        yield c


def test_healthz_returns_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200


def test_version_returns_model_info(client):
    r = client.get("/version")
    assert r.status_code == 200
    assert "model" in r.json()


def test_extract_happy_path(client):
    r = client.post("/extract", json={"text": "Any contract text."})
    assert r.status_code == 200
    body = r.json()
    assert len(body["clauses"]) == 1
    assert body["clauses"][0]["type"] == "Governing Law"


def test_extract_empty_text_is_422(client):
    r = client.post("/extract", json={"text": ""})
    assert r.status_code == 422


def test_extract_bad_clause_type_is_422(client):
    r = client.post("/extract", json={"text": "hello", "clause_types": ["Not A Real Clause"]})
    assert r.status_code == 422


def test_extract_malformed_body_is_422(client):
    r = client.post("/extract", json={})
    assert r.status_code == 422
