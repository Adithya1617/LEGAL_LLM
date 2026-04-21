import json
from pathlib import Path

import pytest

from src.data.validate import validate_jsonl


@pytest.fixture
def good_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "train.jsonl"
    rows = [
        {
            "instruction": "...",
            "input": "Agreement governed by Delaware law.",
            "output": json.dumps({"clauses": [{"type": "Governing Law", "span": "Delaware law."}]}),
        },
        {
            "instruction": "...",
            "input": "nothing interesting here",
            "output": json.dumps({"clauses": []}),
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


@pytest.fixture
def bad_output_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "bad.jsonl"
    rows = [
        {"instruction": "x", "input": "y", "output": "not json"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


@pytest.fixture
def span_not_in_input_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "span_mismatch.jsonl"
    rows = [
        {
            "instruction": "x",
            "input": "completely unrelated text",
            "output": json.dumps({"clauses": [{"type": "Governing Law", "span": "Delaware law."}]}),
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_validate_good_jsonl_reports_no_errors(good_jsonl):
    report = validate_jsonl(good_jsonl)
    assert report.parse_errors == 0
    assert report.schema_errors == 0
    assert report.span_errors == 0
    assert report.negative_ratio == 0.5


def test_validate_reports_parse_errors(bad_output_jsonl):
    report = validate_jsonl(bad_output_jsonl)
    assert report.parse_errors == 1


def test_validate_reports_span_mismatches(span_not_in_input_jsonl):
    report = validate_jsonl(span_not_in_input_jsonl)
    assert report.span_errors == 1
