from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from src.common.schemas import ClauseList


@dataclass
class ValidationReport:
    rows: int = 0
    parse_errors: int = 0
    schema_errors: int = 0
    span_errors: int = 0
    clause_counts: Counter = field(default_factory=Counter)
    negative_rows: int = 0

    @property
    def negative_ratio(self) -> float:
        return self.negative_rows / self.rows if self.rows else 0.0


def validate_jsonl(path: Path) -> ValidationReport:
    report = ValidationReport()
    with path.open() as f:
        for line in f:
            report.rows += 1
            row = json.loads(line)
            try:
                parsed = json.loads(row["output"])
            except json.JSONDecodeError:
                report.parse_errors += 1
                continue
            try:
                cl = ClauseList.model_validate(parsed)
            except ValidationError:
                report.schema_errors += 1
                continue
            if not cl.clauses:
                report.negative_rows += 1
            for c in cl.clauses:
                report.clause_counts[c.type.value] += 1
                if c.span not in row["input"]:
                    report.span_errors += 1
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    for split in ("train", "val", "test"):
        p = args.dir / f"{split}.jsonl"
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        r = validate_jsonl(p)
        print(f"\n=== {split} ===")
        print(
            f"rows: {r.rows}  parse_err: {r.parse_errors}  "
            f"schema_err: {r.schema_errors}  span_err: {r.span_errors}"
        )
        print(f"negative_ratio: {r.negative_ratio:.2%}")
        print("clause counts:")
        for ct, n in sorted(r.clause_counts.items()):
            marker = "  LOW" if n < 20 and split == "train" else ""
            print(f"  {ct}: {n}{marker}")


if __name__ == "__main__":
    main()
