from src.common.prompts import (
    SYSTEM_PROMPT,
    build_few_shot_prompt,
    build_instruction,
    load_few_shot_examples,
)
from src.common.schemas import ClauseType


def test_build_instruction_lists_all_clause_types():
    instr = build_instruction()
    for ct in ClauseType:
        assert ct.value in instr


def test_build_instruction_accepts_subset():
    types = [ClauseType.GOVERNING_LAW, ClauseType.INDEMNIFICATION]
    instr = build_instruction(types)
    assert "Governing Law" in instr
    assert "Indemnification" in instr
    assert "Non-Compete" not in instr


def test_system_prompt_is_nonempty():
    assert len(SYSTEM_PROMPT) > 20


def test_load_few_shot_examples_returns_three():
    examples = load_few_shot_examples()
    assert len(examples) == 3
    for ex in examples:
        assert "input" in ex and "output" in ex


def test_build_few_shot_prompt_contains_all_examples():
    prompt = build_few_shot_prompt("New contract text here.")
    examples = load_few_shot_examples()
    for ex in examples:
        assert ex["input"][:40] in prompt
