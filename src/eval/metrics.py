from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.common.schemas import ClauseList, ClauseType


def json_parse_rate(outputs: list[str]) -> float:
    if not outputs:
        return 0.0
    ok = 0
    for o in outputs:
        try:
            json.loads(o)
            ok += 1
        except json.JSONDecodeError:
            pass
    return ok / len(outputs)


def schema_valid_rate(outputs: list[str]) -> float:
    if not outputs:
        return 0.0
    ok = 0
    for o in outputs:
        try:
            parsed = json.loads(o)
        except json.JSONDecodeError:
            continue
        try:
            ClauseList.model_validate(parsed)
            ok += 1
        except ValidationError:
            pass
    return ok / len(outputs)


@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float


@dataclass
class ClauseMetrics:
    per_type: dict[ClauseType, PRF1] = field(default_factory=dict)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0


def _prf1(tp: int, fp: int, fn: int) -> PRF1:
    # Absent from both gold and prediction → class handled correctly; avoids
    # macro-F1 being dragged down by clauses that happen to be missing in a
    # small test set.
    if tp + fp + fn == 0:
        return PRF1(precision=1.0, recall=1.0, f1=1.0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF1(precision=precision, recall=recall, f1=f1)


def classification_metrics(gold: list[ClauseList], pred: list[ClauseList]) -> ClauseMetrics:
    """Per-chunk, per-clause: is this clause type present?"""
    assert len(gold) == len(pred)
    per: dict[ClauseType, PRF1] = {}
    for ct in ClauseType:
        tp = fp = fn = 0
        for g, p in zip(gold, pred, strict=False):
            g_has = any(c.type == ct for c in g.clauses)
            p_has = any(c.type == ct for c in p.clauses)
            if g_has and p_has:
                tp += 1
            elif p_has and not g_has:
                fp += 1
            elif g_has and not p_has:
                fn += 1
        per[ct] = _prf1(tp, fp, fn)
    macro_p = sum(m.precision for m in per.values()) / len(per)
    macro_r = sum(m.recall for m in per.values()) / len(per)
    macro_f = sum(m.f1 for m in per.values()) / len(per)
    return ClauseMetrics(
        per_type=per,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f,
    )


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def span_token_f1(gold_span: str, pred_span: str) -> float:
    g = set(_tokenize(gold_span))
    p = set(_tokenize(pred_span))
    if not g or not p:
        return 0.0
    common = g & p
    if not common:
        return 0.0
    precision = len(common) / len(p)
    recall = len(common) / len(g)
    return 2 * precision * recall / (precision + recall)


def span_set_f1(gold: list[ClauseList], pred: list[ClauseList]) -> dict[ClauseType, float]:
    """Per-clause-type, averaged over chunks where either side has spans:
    greedy-matched F1 on token-F1 pairs."""
    per: dict[ClauseType, list[float]] = {ct: [] for ct in ClauseType}
    for g, p in zip(gold, pred, strict=False):
        for ct in ClauseType:
            gs = [c.span for c in g.clauses if c.type == ct]
            ps = [c.span for c in p.clauses if c.type == ct]
            if not gs and not ps:
                continue
            if not gs or not ps:
                per[ct].append(0.0)
                continue
            scores: list[float] = []
            used: set[int] = set()
            for gs_i in gs:
                best = 0.0
                best_j = -1
                for pj, ps_j in enumerate(ps):
                    if pj in used:
                        continue
                    s = span_token_f1(gs_i, ps_j)
                    if s > best:
                        best = s
                        best_j = pj
                if best_j >= 0:
                    used.add(best_j)
                scores.append(best)
            per[ct].append(sum(scores) / max(len(gs), len(ps)))
    return {ct: (sum(v) / len(v)) if v else 0.0 for ct, v in per.items()}
