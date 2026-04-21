from src.common.schemas import Clause, ClauseList, ClauseType
from src.eval.metrics import (
    classification_metrics,
    json_parse_rate,
    schema_valid_rate,
    span_token_f1,
)


def test_json_parse_rate():
    outputs = ['{"clauses":[]}', "not json", '{"clauses":[]}']
    assert json_parse_rate(outputs) == 2 / 3


def test_schema_valid_rate():
    outputs = [
        '{"clauses":[]}',
        '{"not_clauses":[]}',
        '{"clauses":[{"type":"Governing Law","span":"x"}]}',
    ]
    assert schema_valid_rate(outputs) == 2 / 3


def test_classification_metrics_all_correct():
    gold = [
        ClauseList(clauses=[Clause(type=ClauseType.GOVERNING_LAW, span="x")]),
        ClauseList(clauses=[]),
    ]
    pred = [
        ClauseList(clauses=[Clause(type=ClauseType.GOVERNING_LAW, span="y")]),
        ClauseList(clauses=[]),
    ]
    m = classification_metrics(gold, pred)
    assert m.macro_f1 == 1.0


def test_classification_metrics_false_positive_drops_precision():
    gold = [ClauseList(clauses=[])]
    pred = [ClauseList(clauses=[Clause(type=ClauseType.GOVERNING_LAW, span="x")])]
    m = classification_metrics(gold, pred)
    gl = m.per_type[ClauseType.GOVERNING_LAW]
    assert gl.precision == 0.0


def test_span_token_f1_exact_match():
    assert span_token_f1("delaware law applies", "delaware law applies") == 1.0


def test_span_token_f1_partial():
    f1 = span_token_f1("delaware law applies here", "delaware law")
    assert 0.5 < f1 < 1.0


def test_span_token_f1_disjoint():
    assert span_token_f1("abc def", "ghi jkl") == 0.0
