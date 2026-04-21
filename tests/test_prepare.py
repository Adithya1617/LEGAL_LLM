from src.common.schemas import ClauseType
from src.data.prepare import (
    CUAD_TO_CLAUSE_TYPE,
    aggregate_gold_spans_in_chunk,
    split_by_contract,
)


def test_cuad_label_map_covers_ten_clauses():
    assert len(CUAD_TO_CLAUSE_TYPE) >= 10
    values = set(CUAD_TO_CLAUSE_TYPE.values())
    assert len(values) == 10  # each of our 10 types is represented at least once
    for ct in ClauseType:
        assert ct in values, f"missing mapping for {ct}"


def test_aggregate_gold_spans_in_chunk_keeps_spans_inside_range():
    gold = [
        {
            "type": ClauseType.GOVERNING_LAW,
            "span": "Delaware law.",
            "start_char": 50,
            "end_char": 63,
        },
        {
            "type": ClauseType.LIABILITY_CAP,
            "span": "cap of $100.",
            "start_char": 200,
            "end_char": 212,
        },
    ]
    chunk_content = "x" * 100
    kept = aggregate_gold_spans_in_chunk(
        gold, chunk_start=0, chunk_end=100, chunk_content=chunk_content
    )
    # First span (50-63) is inside [0,100]; second (200-212) is outside.
    # But first span's text "Delaware law." isn't in chunk_content (all x's), so it's dropped.
    assert kept == []


def test_aggregate_gold_spans_in_chunk_keeps_when_substring_present():
    span_text = "Delaware law."
    chunk_content = "blah blah " + span_text + " more blah"
    start = chunk_content.index(span_text)
    gold = [
        {
            "type": ClauseType.GOVERNING_LAW,
            "span": span_text,
            "start_char": start,
            "end_char": start + len(span_text),
        }
    ]
    kept = aggregate_gold_spans_in_chunk(
        gold, chunk_start=0, chunk_end=len(chunk_content), chunk_content=chunk_content
    )
    assert len(kept) == 1
    assert kept[0]["type"] == ClauseType.GOVERNING_LAW


def test_split_by_contract_has_no_leakage():
    contracts = [f"c{i}" for i in range(100)]
    splits = split_by_contract(contracts, seed=42)
    train, val, test = splits["train"], splits["val"], splits["test"]
    all_ids = set(train) | set(val) | set(test)
    assert len(all_ids) == 100
    assert len(train) + len(val) + len(test) == 100


def test_split_by_contract_ratio_approx_80_10_10():
    contracts = [f"c{i}" for i in range(100)]
    splits = split_by_contract(contracts, seed=42)
    assert 75 <= len(splits["train"]) <= 85
    assert 5 <= len(splits["val"]) <= 15
    assert 5 <= len(splits["test"]) <= 15
