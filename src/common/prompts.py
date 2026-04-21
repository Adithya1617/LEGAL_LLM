from __future__ import annotations

import json
from pathlib import Path

from src.common.schemas import ClauseType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEW_SHOT_PATH = _REPO_ROOT / "configs" / "few_shot_examples.json"

SYSTEM_PROMPT = (
    "You are a legal-document analysis assistant. "
    "You extract specific clause types from contract excerpts and return them as JSON. "
    "Extract only spans that appear verbatim in the input. "
    "If a clause type is not present, do not include it in the output."
)


def build_instruction(clause_types: list[ClauseType] | None = None) -> str:
    types = clause_types if clause_types is not None else list(ClauseType)
    type_list = ", ".join(t.value for t in types)
    return (
        "Extract the following clause types from the contract excerpt below. "
        "Return a JSON object with a 'clauses' array. Each element must have "
        "'type' (one of the listed clause types) and 'span' (the exact text "
        "from the excerpt).\n"
        f"Clause types: [{type_list}]\n\n"
        'If no listed clause appears in the excerpt, return {"clauses": []}.'
    )


def load_few_shot_examples() -> list[dict]:
    return json.loads(_FEW_SHOT_PATH.read_text())


def build_few_shot_prompt(contract_text: str) -> str:
    examples = load_few_shot_examples()
    instr = build_instruction()
    parts = [instr, ""]
    for ex in examples:
        parts.append(f"Excerpt:\n{ex['input']}")
        parts.append(f"Output:\n{json.dumps(ex['output'])}")
        parts.append("")
    parts.append(f"Excerpt:\n{contract_text}")
    parts.append("Output:")
    return "\n".join(parts)
