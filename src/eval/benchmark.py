"""Run all 4 providers, emit eval_results.md with comparison tables."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROVIDERS = ["ft", "base-zs", "base-fs", "gpt-4o-mini"]

# USD per 1K output tokens (update if pricing changes).
API_COST_PER_1K_OUT = {"gpt-4o-mini": 0.0006}


def _run_one(provider: str) -> dict:
    out = Path(f"outputs/eval/{provider}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "python",
            "-m",
            "src.eval.evaluate",
            "--provider",
            provider,
            "--out",
            str(out),
        ]
    )
    return json.loads(out.read_text())


def _cost_per_1k_contracts(provider: str, avg_out_tokens: float = 200.0) -> float:
    rate = API_COST_PER_1K_OUT.get(provider)
    if rate is None:
        return 0.0
    return rate * (avg_out_tokens / 1000) * 1000


def _markdown_table(reports: dict[str, dict]) -> str:
    lines = [
        "| Model | Macro F1 | Macro P | Macro R | JSON % | Schema % | p50 (ms) | p95 (ms) | $/1K contracts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in PROVIDERS:
        r = reports[p]
        cost = _cost_per_1k_contracts(p)
        lines.append(
            f"| {p} | {r['macro_f1']:.3f} | {r['macro_precision']:.3f} | "
            f"{r['macro_recall']:.3f} | {r['json_parse_rate'] * 100:.1f}% | "
            f"{r['schema_valid_rate'] * 100:.1f}% | {r['p50_latency_ms']:.0f} | "
            f"{r['p95_latency_ms']:.0f} | ${cost:.4f} |"
        )
    return "\n".join(lines)


def _per_type_table(reports: dict[str, dict]) -> str:
    ft = reports["ft"]
    lines = [
        "| Clause | FT F1 | FT P | FT R | FT Span F1 |",
        "|---|---|---|---|---|",
    ]
    for ct, m in sorted(ft["per_type"].items()):
        lines.append(
            f"| {ct} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['span_f1']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    reports = {p: _run_one(p) for p in PROVIDERS}
    md = [
        "# Evaluation Results",
        "",
        "## Overall (4-model comparison)",
        "",
        _markdown_table(reports),
        "",
        "## Per-clause breakdown (fine-tuned model)",
        "",
        _per_type_table(reports),
        "",
        "## Notes",
        "- Fine-tuned model = Llama-3.2-3B + QLoRA on CUAD train split, merged to fp16.",
        "- `base-zs` = base Llama-3.2-3B-Instruct, zero-shot.",
        "- `base-fs` = base Llama-3.2-3B-Instruct, 3 fixed in-context examples.",
        "- `gpt-4o-mini` = OpenAI API, same 3 in-context examples as base-fs.",
        "- Latencies measured on RTX 3060 6GB at batch size 1.",
        "- `$/1K contracts` assumes ~200 output tokens per contract chunk.",
    ]
    Path("eval_results.md").write_text("\n".join(md))
    print("Wrote eval_results.md")


if __name__ == "__main__":
    main()
