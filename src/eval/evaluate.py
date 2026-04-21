"""Run one provider over the test JSONL and compute all metrics."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.common.prompts import build_few_shot_prompt, build_instruction
from src.common.schemas import ClauseList
from src.eval.metrics import (
    classification_metrics,
    json_parse_rate,
    schema_valid_rate,
    span_set_f1,
)
from src.eval.providers import LocalHFProvider, OpenAIProvider, Provider


@dataclass
class EvalReport:
    provider: str
    n: int
    json_parse_rate: float
    schema_valid_rate: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_type: dict
    p50_latency_ms: float
    p95_latency_ms: float


def evaluate(provider: Provider, test_jsonl: Path, few_shot: bool) -> EvalReport:
    gold: list[ClauseList] = []
    pred: list[ClauseList] = []
    raw_outputs: list[str] = []
    latencies_ms: list[float] = []

    with test_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            if few_shot:
                prompt = build_few_shot_prompt(row["input"])
            else:
                prompt = f"{build_instruction()}\n\nExcerpt:\n{row['input']}"
            t0 = time.perf_counter()
            raw = provider.generate(prompt)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            raw_outputs.append(raw)
            gold.append(ClauseList.model_validate_json(row["output"]))
            try:
                pred.append(ClauseList.model_validate_json(raw))
            except Exception:
                pred.append(ClauseList(clauses=[]))

    clf = classification_metrics(gold, pred)
    span_f1 = span_set_f1(gold, pred)
    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]

    per = {
        ct.value: {
            "precision": clf.per_type[ct].precision,
            "recall": clf.per_type[ct].recall,
            "f1": clf.per_type[ct].f1,
            "span_f1": span_f1[ct],
        }
        for ct in clf.per_type
    }

    return EvalReport(
        provider=provider.name,
        n=len(gold),
        json_parse_rate=json_parse_rate(raw_outputs),
        schema_valid_rate=schema_valid_rate(raw_outputs),
        macro_f1=clf.macro_f1,
        macro_precision=clf.macro_precision,
        macro_recall=clf.macro_recall,
        per_type=per,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--provider",
        required=True,
        choices=["ft", "base-zs", "base-fs", "gpt-4o-mini"],
    )
    ap.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prov: Provider
    few_shot = False
    if args.provider == "ft":
        prov = LocalHFProvider(name="ft", model_path="models/merged")
    elif args.provider == "base-zs":
        prov = LocalHFProvider(name="base-zs", model_path="unsloth/Llama-3.2-3B-Instruct")
    elif args.provider == "base-fs":
        prov = LocalHFProvider(name="base-fs", model_path="unsloth/Llama-3.2-3B-Instruct")
        few_shot = True
    else:
        prov = OpenAIProvider(name="gpt-4o-mini")
        few_shot = True

    report = evaluate(prov, args.test, few_shot=few_shot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2))
    print(f"Saved {args.out}: macro_f1={report.macro_f1:.3f}")


if __name__ == "__main__":
    main()
