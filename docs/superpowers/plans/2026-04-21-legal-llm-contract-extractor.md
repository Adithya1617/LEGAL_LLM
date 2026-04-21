# Legal LLM — Contract Clause Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Llama-3.2-3B with QLoRA on CUAD to extract 10 commercial-contract clause types as structured JSON, serve it via FastAPI with a Gradio PDF-upload demo, publish as a Dockerized image with CI.

**Architecture:** Data (CUAD → pydantic-validated JSONL) → Training (Unsloth + TRL SFTTrainer + W&B) → LoRA merge → Evaluation (4-way benchmark vs base/few-shot/GPT-4o-mini + latency/cost) → FastAPI with Outlines constrained JSON decoding → Gradio demo calling API over HTTP with pypdf ingestion → Multi-stage Docker image baked with merged model → GitHub Actions CI (lint + pytest + mocked API tests + Docker build on tag → ghcr.io).

**Tech Stack:** Python 3.11, PyTorch 2.x (CUDA 12.1), Unsloth, TRL, PEFT, bitsandbytes, transformers, datasets, Outlines, pydantic v2, FastAPI, Gradio, pypdf, httpx, pytest, ruff, mypy, structlog, Weights & Biases, Docker (multi-stage), GitHub Actions.

**Reference spec:** `docs/superpowers/specs/2026-04-21-legal-llm-contract-extractor-design.md`

---

## File Structure

```
src/
  common/
    __init__.py
    schemas.py              # Pydantic models: Clause, ClauseList, ClauseType, ExtractRequest, ExtractResponse
    chunking.py             # Token-aware chunker used by both prepare.py and app.py
    prompts.py              # Prompt templates (instruction text, few-shot exemplars)
  data/
    __init__.py
    prepare.py              # CUAD → chunked JSONL with 10 clause types + negatives + contract-level split
    validate.py             # Pydantic validation + span-in-text check + count reporting
  train/
    __init__.py
    train.py                # Unsloth + TRL SFTTrainer, W&B logged
    merge.py                # LoRA merge → fp16 model dir
  eval/
    __init__.py
    metrics.py              # JSON/schema/classification/span metrics
    evaluate.py             # Run one model over test set → metrics dict
    benchmark.py            # Orchestrate 4 models + latency/cost → eval_results.md
    providers.py            # Uniform model-call interface: local HF, few-shot wrapper, OpenAI
  serve/
    __init__.py
    api.py                  # FastAPI app, Outlines constrained generation, structlog
    app.py                  # Gradio UI, pypdf ingestion, HTTP to FastAPI
    extractor.py            # Model-backed extraction logic (imported by api.py; mocked in tests)
configs/
  llama32_qlora.yaml        # (rename from phi3_qlora.yaml)
  serve.yaml
  few_shot_examples.json    # 3 fixed exemplars for few-shot baselines
notebooks/
  01_data_exploration.ipynb
tests/
  conftest.py               # Shared fixtures
  test_schemas.py           # Pydantic schema unit tests
  test_chunking.py          # Chunker unit tests
  test_prepare.py           # Data prep unit tests (fixtures, not full CUAD)
  test_validate.py
  test_metrics.py
  test_api.py               # FastAPI tests with mocked extractor
  test_api.http             # Manual REST Client calls
docker/
  Dockerfile                # Multi-stage: builder → model-fetcher → runtime
  docker-compose.yml        # api + demo services
.github/workflows/
  ci.yml                    # lint + test + docker build (builder stage only)
  eval.yml                  # release: full image build + ghcr.io publish
pyproject.toml              # ruff + mypy + pytest config
requirements.txt            # pinned runtime deps
requirements-dev.txt        # ruff, mypy, pytest, pre-commit
.pre-commit-config.yaml
Makefile                    # data, train, eval, serve, docker-build, test, lint
README.md
eval_results.md             # Written by benchmark.py
```

**Design units:**
- **`src/common/`** is the interface layer — pydantic schemas, chunking, prompts. Everything downstream imports from here. Zero external deps beyond pydantic + tokenizer.
- **`src/data/`** is a one-shot pipeline. No state, no classes, just pure functions.
- **`src/train/`** owns the LoRA training loop; depends on `common/` only.
- **`src/eval/`** is organized so `metrics.py` is unit-testable without a model, `providers.py` abstracts the model-call surface, and `evaluate.py`/`benchmark.py` compose them.
- **`src/serve/extractor.py`** is the seam between FastAPI and the model — tests mock this one function, keeping route tests CPU-fast.

---

## Task 0: Project setup — git, deps, tooling

**Files:**
- Create: `pyproject.toml`, `.pre-commit-config.yaml`, `requirements-dev.txt`
- Modify: `.gitignore` (add logs, `eval_results.md` NOT ignored)
- Modify: `Makefile`

- [ ] **Step 0.1: Initialize git repository**

```bash
git init
git add .
git commit -m "chore: initial scaffold"
```

- [ ] **Step 0.2: Rename Phi-3 config placeholder to Llama config**

```bash
git mv configs/phi3_qlora.yaml configs/llama32_qlora.yaml
git commit -m "chore: rename training config from phi3 to llama32"
```

- [ ] **Step 0.3: Create `pyproject.toml` with ruff + mypy + pytest config**

```toml
[project]
name = "legal-llm"
version = "0.1.0"
requires-python = ">=3.11,<3.12"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]  # line length is handled by formatter

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true  # unsloth/outlines lack stubs

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
filterwarnings = ["ignore::DeprecationWarning"]
```

- [ ] **Step 0.4: Create `requirements-dev.txt`**

```
ruff==0.6.9
mypy==1.11.2
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
pre-commit==4.0.1
```

- [ ] **Step 0.5: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: check-merge-conflict
      - id: end-of-file-fixer
      - id: trailing-whitespace
```

- [ ] **Step 0.6: Install dev deps and pre-commit**

```bash
source .venv/bin/activate  # or activate however your venv works
pip install -r requirements-dev.txt
pre-commit install
```

- [ ] **Step 0.7: Update `.gitignore` — keep `eval_results.md` tracked, ignore logs and W&B**

Edit `.gitignore` to ensure it contains (it already contains most of these — confirm `eval_results.md` is NOT ignored, and add these lines if missing):

```
# already present: .env .venv/ venv/ data/ models/ __pycache__/ *.gguf *.bin *.pt *.safetensors wandb/ .ipynb_checkpoints/

# add these:
logs/
*.log
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
outputs/
```

- [ ] **Step 0.8: Populate `Makefile`**

```make
.PHONY: data train eval serve demo docker-build test lint format ci

data:
	python -m src.data.prepare && python -m src.data.validate

train:
	python -m src.train.train --config configs/llama32_qlora.yaml

merge:
	python -m src.train.merge --adapter models/checkpoints/best --out models/merged

eval:
	python -m src.eval.benchmark

serve:
	uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 --reload

demo:
	python -m src.serve.app

docker-build:
	docker build -f docker/Dockerfile -t legal-llm:local .

test:
	pytest tests/

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

ci: lint test
```

- [ ] **Step 0.9: Commit setup**

```bash
git add pyproject.toml requirements-dev.txt .pre-commit-config.yaml .gitignore Makefile
git commit -m "chore: add dev tooling (ruff, mypy, pytest, pre-commit, make)"
```

---

## Task 1: Pydantic schemas in `src/common/schemas.py`

**Files:**
- Create: `src/common/__init__.py` (empty)
- Create: `src/common/schemas.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1.1: Write failing test for `ClauseType` enum and `Clause` model**

Create `tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError

from src.common.schemas import (
    Clause,
    ClauseList,
    ClauseType,
    ExtractRequest,
    ExtractResponse,
)


def test_clause_type_has_exactly_ten_members():
    assert len(list(ClauseType)) == 10


def test_clause_type_contains_expected_values():
    values = {c.value for c in ClauseType}
    assert "Governing Law" in values
    assert "Indemnification" in values
    assert "Auto-Renewal" in values


def test_clause_accepts_valid_type_and_span():
    c = Clause(type=ClauseType.GOVERNING_LAW, span="This Agreement is governed by Delaware law.")
    assert c.span.startswith("This Agreement")


def test_clause_rejects_empty_span():
    with pytest.raises(ValidationError):
        Clause(type=ClauseType.GOVERNING_LAW, span="")


def test_clause_list_allows_empty():
    cl = ClauseList(clauses=[])
    assert cl.clauses == []


def test_clause_list_round_trips_json():
    payload = '{"clauses":[{"type":"Governing Law","span":"x"}]}'
    cl = ClauseList.model_validate_json(payload)
    assert cl.clauses[0].type == ClauseType.GOVERNING_LAW
    assert cl.model_dump_json() == payload


def test_extract_request_defaults_clause_types_to_none():
    req = ExtractRequest(text="hello")
    assert req.clause_types is None


def test_extract_response_fields():
    resp = ExtractResponse(
        clauses=[],
        latency_ms=12.5,
        model_version="llama-3.2-3b-legal@abc1234",
    )
    assert resp.latency_ms == 12.5
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
pytest tests/test_schemas.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.common.schemas'`

- [ ] **Step 1.3: Create `src/common/__init__.py` (empty file)**

```bash
touch src/common/__init__.py tests/__init__.py
```

- [ ] **Step 1.4: Implement `src/common/schemas.py`**

```python
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    GOVERNING_LAW = "Governing Law"
    INDEMNIFICATION = "Indemnification"
    NON_COMPETE = "Non-Compete"
    TERMINATION_FOR_CONVENIENCE = "Termination for Convenience"
    LIABILITY_CAP = "Liability Cap"
    EXCLUSIVITY = "Exclusivity"
    IP_ASSIGNMENT = "IP Assignment"
    CONFIDENTIALITY = "Confidentiality"
    CHANGE_OF_CONTROL = "Change of Control"
    AUTO_RENEWAL = "Auto-Renewal"


class Clause(BaseModel):
    type: ClauseType
    span: str = Field(min_length=1)


class ClauseList(BaseModel):
    clauses: list[Clause] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)
    clause_types: list[ClauseType] | None = None


class ExtractResponse(BaseModel):
    clauses: list[Clause]
    latency_ms: float
    model_version: str
```

- [ ] **Step 1.5: Create `tests/conftest.py` (empty for now, we'll add fixtures later)**

```python
# Fixtures added as tasks need them.
```

- [ ] **Step 1.6: Run tests — must pass**

```bash
pytest tests/test_schemas.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 1.7: Commit**

```bash
git add src/common tests/__init__.py tests/conftest.py tests/test_schemas.py
git commit -m "feat(common): pydantic schemas for clauses and API payloads"
```

---

## Task 2: Token-aware chunker in `src/common/chunking.py`

**Files:**
- Create: `src/common/chunking.py`
- Create: `tests/test_chunking.py`

Design: single function `chunk_text(text, tokenizer, chunk_tokens, overlap_tokens) -> list[Chunk]` where each `Chunk` carries `text`, `start_char`, `end_char`. Char offsets matter because we need to check later whether a gold span falls inside a chunk.

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_chunking.py`:

```python
from transformers import AutoTokenizer

from src.common.chunking import Chunk, chunk_text

TOKENIZER_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def _tok():
    # gated model — fall back to a public tokenizer with same BPE family if needed
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception:
        return AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-3B")


def test_chunk_short_text_returns_single_chunk():
    tok = _tok()
    text = "This is a short clause."
    chunks = chunk_text(text, tok, chunk_tokens=1500, overlap_tokens=200)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)


def test_chunk_long_text_splits_with_overlap():
    tok = _tok()
    text = ("A clause. " * 1000).strip()
    chunks = chunk_text(text, tok, chunk_tokens=500, overlap_tokens=50)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        assert chunks[i].end_char > chunks[i + 1].start_char, "chunks must overlap"


def test_chunks_cover_full_text():
    tok = _tok()
    text = ("Sentence. " * 500).strip()
    chunks = chunk_text(text, tok, chunk_tokens=500, overlap_tokens=50)
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)


def test_chunk_offsets_are_substrings():
    tok = _tok()
    text = ("Sentence. " * 500).strip()
    chunks = chunk_text(text, tok, chunk_tokens=500, overlap_tokens=50)
    for c in chunks:
        assert text[c.start_char:c.end_char] == c.text
```

- [ ] **Step 2.2: Run — must fail**

```bash
pytest tests/test_chunking.py -v
```
Expected: FAIL (no module).

- [ ] **Step 2.3: Implement chunker**

Create `src/common/chunking.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    text: str
    start_char: int
    end_char: int


class _Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def chunk_text(
    text: str,
    tokenizer: _Tokenizer,
    chunk_tokens: int = 1500,
    overlap_tokens: int = 200,
) -> list[Chunk]:
    """
    Split `text` into overlapping chunks of at most `chunk_tokens` tokens.

    We tokenize with offsets so we can recover exact character boundaries per
    chunk. This matters downstream: we need to check whether a gold span from
    CUAD falls inside a chunk's char range before including that span as a
    training label for the chunk.
    """
    if chunk_tokens <= overlap_tokens:
        raise ValueError("chunk_tokens must be greater than overlap_tokens")

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if len(input_ids) <= chunk_tokens:
        return [Chunk(text=text, start_char=0, end_char=len(text))]

    stride = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    i = 0
    while i < len(input_ids):
        end = min(i + chunk_tokens, len(input_ids))
        start_char = offsets[i][0]
        end_char = offsets[end - 1][1]
        chunks.append(Chunk(text=text[start_char:end_char], start_char=start_char, end_char=end_char))
        if end == len(input_ids):
            break
        i += stride
    return chunks
```

- [ ] **Step 2.4: Run — must pass**

```bash
pytest tests/test_chunking.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/common/chunking.py tests/test_chunking.py
git commit -m "feat(common): token-aware chunker with char offsets"
```

---

## Task 3: Prompt templates in `src/common/prompts.py`

**Files:**
- Create: `src/common/prompts.py`
- Create: `configs/few_shot_examples.json`
- Create: `tests/test_prompts.py`

The instruction text must be identical between training and inference. Centralizing it here is the reason this file exists.

- [ ] **Step 3.1: Write failing test**

Create `tests/test_prompts.py`:

```python
from src.common.prompts import (
    SYSTEM_PROMPT,
    build_instruction,
    build_few_shot_prompt,
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
```

- [ ] **Step 3.2: Run — must fail**

```bash
pytest tests/test_prompts.py -v
```

- [ ] **Step 3.3: Create `configs/few_shot_examples.json`**

```json
[
  {
    "input": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles. Each party's total liability under this Agreement shall not exceed the total fees paid in the twelve months preceding the claim.",
    "output": {
      "clauses": [
        {"type": "Governing Law", "span": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles."},
        {"type": "Liability Cap", "span": "Each party's total liability under this Agreement shall not exceed the total fees paid in the twelve months preceding the claim."}
      ]
    }
  },
  {
    "input": "Each party shall maintain the confidentiality of the other party's Confidential Information and shall not disclose it to any third party without prior written consent. All intellectual property created by Contractor in the performance of this Agreement shall be the sole and exclusive property of the Company.",
    "output": {
      "clauses": [
        {"type": "Confidentiality", "span": "Each party shall maintain the confidentiality of the other party's Confidential Information and shall not disclose it to any third party without prior written consent."},
        {"type": "IP Assignment", "span": "All intellectual property created by Contractor in the performance of this Agreement shall be the sole and exclusive property of the Company."}
      ]
    }
  },
  {
    "input": "The initial term of this Agreement is twelve (12) months. No automatic renewal shall apply and any extension requires mutual written consent. This section sets forth the payment schedule and invoicing terms between the parties.",
    "output": {
      "clauses": []
    }
  }
]
```

- [ ] **Step 3.4: Implement `src/common/prompts.py`**

```python
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
        "If no listed clause appears in the excerpt, return {\"clauses\": []}."
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
```

- [ ] **Step 3.5: Run — must pass**

```bash
pytest tests/test_prompts.py -v
```

- [ ] **Step 3.6: Commit**

```bash
git add src/common/prompts.py configs/few_shot_examples.json tests/test_prompts.py
git commit -m "feat(common): prompt templates and few-shot exemplars"
```

---

## Task 4: Install ML runtime dependencies

**Files:**
- Modify: `requirements.txt` (append ML packages not yet pinned)

`pip freeze` already captured your venv. This task installs the ML stack we'll need. After install, we re-freeze to lock versions.

- [ ] **Step 4.1: Install Unsloth + ML stack**

The Unsloth install command is version- and CUDA-specific. For CUDA 12.1 + PyTorch 2.4:

```bash
source .venv/bin/activate
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install \
    "transformers==4.45.2" \
    "datasets==3.0.1" \
    "peft==0.13.2" \
    "trl==0.11.4" \
    "bitsandbytes==0.44.1" \
    "accelerate==1.0.1" \
    "wandb==0.18.3" \
    "outlines==0.1.1" \
    "pydantic==2.9.2" \
    "fastapi==0.115.0" \
    "uvicorn[standard]==0.31.1" \
    "gradio==4.44.1" \
    "pypdf==5.0.1" \
    "structlog==24.4.0" \
    "httpx==0.27.2" \
    "openai==1.51.2" \
    "tiktoken==0.8.0" \
    "pyyaml==6.0.2" \
    "python-multipart==0.0.12"
```

- [ ] **Step 4.2: Verify GPU is visible**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `True NVIDIA GeForce RTX 3060`

- [ ] **Step 4.3: Re-freeze requirements**

```bash
pip freeze > requirements.txt
```

- [ ] **Step 4.4: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin ML runtime dependencies"
```

---

## Task 5: CUAD dataset loader + chunked sample generation in `src/data/prepare.py`

**Files:**
- Create: `src/data/__init__.py` (empty)
- Create: `src/data/prepare.py`
- Create: `tests/test_prepare.py`

This is the heaviest data task. CUAD's native format is SQuAD-style (per-clause, per-contract questions with highlighted answer spans). We transform to our all-at-once JSONL.

- [ ] **Step 5.1: Create `src/data/__init__.py`**

```bash
touch src/data/__init__.py
```

- [ ] **Step 5.2: Write failing test using a mini fixture**

Create `tests/test_prepare.py`:

```python
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
        {"type": ClauseType.GOVERNING_LAW, "span": "Delaware law.", "start_char": 50, "end_char": 63},
        {"type": ClauseType.LIABILITY_CAP, "span": "cap of $100.", "start_char": 200, "end_char": 212},
    ]
    chunk_content = "x" * 100
    kept = aggregate_gold_spans_in_chunk(gold, chunk_start=0, chunk_end=100, chunk_content=chunk_content)
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
    kept = aggregate_gold_spans_in_chunk(gold, chunk_start=0, chunk_end=len(chunk_content), chunk_content=chunk_content)
    assert len(kept) == 1
    assert kept[0]["type"] == ClauseType.GOVERNING_LAW


def test_split_by_contract_has_no_leakage():
    contracts = [f"c{i}" for i in range(100)]
    splits = split_by_contract(contracts, seed=42)
    train, val, test = splits["train"], splits["val"], splits["test"]
    all_ids = set(train) | set(val) | set(test)
    assert len(all_ids) == 100  # no duplicates across splits
    assert len(train) + len(val) + len(test) == 100


def test_split_by_contract_ratio_approx_80_10_10():
    contracts = [f"c{i}" for i in range(100)]
    splits = split_by_contract(contracts, seed=42)
    assert 75 <= len(splits["train"]) <= 85
    assert 5 <= len(splits["val"]) <= 15
    assert 5 <= len(splits["test"]) <= 15
```

- [ ] **Step 5.3: Run — must fail**

```bash
pytest tests/test_prepare.py -v
```

- [ ] **Step 5.4: Implement `src/data/prepare.py`**

```python
"""
Convert CUAD (SQuAD-style QA) into all-clauses-at-once instruction JSONL.

CUAD has 41 clause-type "questions" per contract (e.g., "Highlight the parts
(if any) of this contract related to 'Governing Law'..."). We keep 10.

For each contract: concatenate all its paragraphs, chunk into ~1500-token
windows with 200-token overlap, and for each chunk produce one training
example whose gold output is the JSON array of clauses whose spans fall
inside the chunk.
"""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from src.common.chunking import chunk_text
from src.common.prompts import build_instruction
from src.common.schemas import ClauseList, ClauseType

# --- Mapping from CUAD question labels to our 10 clause types ----------------
# CUAD question labels are long and specific ("Governing Law", "Non-Compete",
# "Cap on Liability", etc.). We canonicalize to ClauseType.
CUAD_TO_CLAUSE_TYPE: dict[str, ClauseType] = {
    "Governing Law": ClauseType.GOVERNING_LAW,
    "Indemnification": ClauseType.INDEMNIFICATION,
    "Non-Compete": ClauseType.NON_COMPETE,
    "Termination For Convenience": ClauseType.TERMINATION_FOR_CONVENIENCE,
    "Cap On Liability": ClauseType.LIABILITY_CAP,
    "Exclusivity": ClauseType.EXCLUSIVITY,
    "Ip Ownership Assignment": ClauseType.IP_ASSIGNMENT,
    # Confidentiality doesn't have a single CUAD label — we use:
    "Non-Disparagement": ClauseType.CONFIDENTIALITY,  # placeholder, refined in validate
    "Change Of Control": ClauseType.CHANGE_OF_CONTROL,
    "Renewal Term": ClauseType.AUTO_RENEWAL,
}
# NOTE: The notebook `01_data_exploration.ipynb` dumps exact CUAD category
# strings — revisit this mapping once those are inspected. If CUAD uses
# "Audit Rights" or something different for confidentiality, update here.

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def aggregate_gold_spans_in_chunk(
    gold: list[dict], chunk_start: int, chunk_end: int, chunk_content: str
) -> list[dict]:
    """Keep gold spans whose char range falls inside [chunk_start, chunk_end]
    AND whose span text appears verbatim in chunk_content."""
    kept: list[dict] = []
    for g in gold:
        if g["start_char"] >= chunk_start and g["end_char"] <= chunk_end:
            if g["span"] in chunk_content:
                kept.append(g)
    return kept


def split_by_contract(contract_ids: list[str], seed: int = 42) -> dict[str, list[str]]:
    rng = random.Random(seed)
    ids = list(contract_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def _cuad_to_contracts() -> dict[str, list[dict]]:
    """
    Load CUAD-v1 and regroup by contract.

    Returns: {contract_id: [{type, span, start_char, end_char}, ...]}

    Also returns the per-contract full text via a second dict.
    """
    ds = load_dataset("theatticusproject/cuad-qa", split="train")
    # CUAD's "context" is the full contract text (repeated across the 41 Q's).
    # `answers` gives answer_start and text for the gold spans.
    gold_by_contract: dict[str, list[dict]] = {}
    text_by_contract: dict[str, str] = {}
    for row in ds:
        cid = row["id"].split("_")[0]  # e.g., "CONTRACT123_Q0" -> "CONTRACT123"
        text_by_contract[cid] = row["context"]
        # Infer CUAD category from the question text.
        category = row["question"].split('"')[1] if '"' in row["question"] else row["question"]
        ctype = CUAD_TO_CLAUSE_TYPE.get(category)
        if ctype is None:
            continue
        gold_by_contract.setdefault(cid, [])
        for start, text in zip(row["answers"]["answer_start"], row["answers"]["text"]):
            gold_by_contract[cid].append(
                {
                    "type": ctype,
                    "span": text,
                    "start_char": start,
                    "end_char": start + len(text),
                }
            )
    return {"gold": gold_by_contract, "text": text_by_contract}


def _build_row(instruction: str, chunk_content: str, kept: list[dict]) -> dict:
    clause_list = ClauseList(
        clauses=[{"type": g["type"], "span": g["span"]} for g in kept]
    )
    return {
        "instruction": instruction,
        "input": chunk_content,
        "output": clause_list.model_dump_json(),
    }


def prepare(
    out_dir: Path,
    seed: int = 42,
    chunk_tokens: int = 1500,
    overlap_tokens: int = 200,
    negative_keep_prob: float = 0.30,
) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    data = _cuad_to_contracts()
    gold_by_c = data["gold"]
    text_by_c = data["text"]
    contract_ids = sorted(text_by_c.keys())
    splits = split_by_contract(contract_ids, seed=seed)
    instruction = build_instruction()

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, cids in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with out_path.open("w") as f:
            for cid in cids:
                text = text_by_c[cid]
                gold = gold_by_c.get(cid, [])
                chunks = chunk_text(text, tok, chunk_tokens, overlap_tokens)
                for c in chunks:
                    kept = aggregate_gold_spans_in_chunk(
                        gold, c.start_char, c.end_char, c.text
                    )
                    if not kept:
                        # Keep only negative_keep_prob fraction of negatives
                        # in train; always keep in val/test for full coverage.
                        if split_name == "train" and rng.random() > negative_keep_prob:
                            continue
                    row = _build_row(instruction, c.text, kept)
                    row["_meta"] = {"contract_id": cid, "split": split_name}
                    f.write(json.dumps(row) + "\n")
        print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare(args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.5: Run unit tests — pure-function tests must pass**

```bash
pytest tests/test_prepare.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5.6: Run the full pipeline end-to-end on CUAD (this downloads ~400MB)**

```bash
python -m src.data.prepare
wc -l data/processed/*.jsonl
```
Expected: three non-empty JSONL files.

- [ ] **Step 5.7: Commit**

```bash
git add src/data/prepare.py src/data/__init__.py tests/test_prepare.py
git commit -m "feat(data): CUAD → chunked JSONL with contract-level splits"
```

---

## Task 6: Data validation in `src/data/validate.py`

**Files:**
- Create: `src/data/validate.py`
- Create: `tests/test_validate.py`

- [ ] **Step 6.1: Write failing test**

Create `tests/test_validate.py`:

```python
import json
from pathlib import Path

import pytest

from src.data.validate import ValidationReport, validate_jsonl


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
```

- [ ] **Step 6.2: Run — must fail**

```bash
pytest tests/test_validate.py -v
```

- [ ] **Step 6.3: Implement `src/data/validate.py`**

```python
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
                cl = ClauseList.model_validate_json(row["output"])
            except json.JSONDecodeError:
                report.parse_errors += 1
                continue
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
        print(f"rows: {r.rows}  parse_err: {r.parse_errors}  "
              f"schema_err: {r.schema_errors}  span_err: {r.span_errors}")
        print(f"negative_ratio: {r.negative_ratio:.2%}")
        print("clause counts:")
        for ct, n in sorted(r.clause_counts.items()):
            marker = "  ⚠ LOW" if n < 20 and split == "train" else ""
            print(f"  {ct}: {n}{marker}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run — must pass**

```bash
pytest tests/test_validate.py -v
python -m src.data.validate
```

- [ ] **Step 6.5: Commit**

```bash
git add src/data/validate.py tests/test_validate.py
git commit -m "feat(data): pydantic + span-in-text validator with per-clause counts"
```

---

## Task 7: Data exploration notebook

**Files:**
- Modify: `notebooks/01_data_exploration.ipynb`

- [ ] **Step 7.1: Populate the notebook with a 4-cell exploration**

Replace the empty notebook at `notebooks/01_data_exploration.ipynb` with a notebook containing these cells (use Jupyter or write JSON directly):

**Cell 1 — Load CUAD and list all question categories**:
```python
from datasets import load_dataset
ds = load_dataset("theatticusproject/cuad-qa", split="train")
import collections
cats = collections.Counter()
for row in ds:
    q = row["question"]
    cat = q.split('"')[1] if '"' in q else q
    cats[cat] += 1
for cat, n in cats.most_common():
    print(f"{n:6d}  {cat}")
```

**Cell 2 — Per-category count of contracts with at least one span**:
```python
cat_to_contracts_with_span = collections.defaultdict(set)
for row in ds:
    if len(row["answers"]["text"]) > 0:
        cid = row["id"].split("_")[0]
        cat = row["question"].split('"')[1] if '"' in row["question"] else row["question"]
        cat_to_contracts_with_span[cat].add(cid)
for cat, contracts in sorted(cat_to_contracts_with_span.items(), key=lambda kv: -len(kv[1])):
    print(f"{len(contracts):4d}  {cat}")
```

**Cell 3 — Verify our 10-clause mapping**:
```python
from src.data.prepare import CUAD_TO_CLAUSE_TYPE
missing = [c for c in CUAD_TO_CLAUSE_TYPE if c not in cats]
print("Missing CUAD categories from our map:", missing)
```

**Cell 4 — Run validation on prepared data**:
```python
from pathlib import Path
from src.data.validate import validate_jsonl
for split in ("train", "val", "test"):
    p = Path("data/processed") / f"{split}.jsonl"
    r = validate_jsonl(p)
    print(split, r)
```

- [ ] **Step 7.2: Run all cells. If Cell 3 prints any missing categories, update `CUAD_TO_CLAUSE_TYPE` in `src/data/prepare.py` to use real CUAD category strings. Re-run `python -m src.data.prepare` and re-validate.**

- [ ] **Step 7.3: Commit (clear outputs before committing)**

```bash
jupyter nbconvert --clear-output --inplace notebooks/01_data_exploration.ipynb
git add notebooks/01_data_exploration.ipynb
# If CUAD_TO_CLAUSE_TYPE was updated:
git add src/data/prepare.py
git commit -m "feat(notebook): CUAD data exploration; verify clause-type mapping"
```

---

## Task 8: Training config YAML

**Files:**
- Modify: `configs/llama32_qlora.yaml`

- [ ] **Step 8.1: Populate the config**

```yaml
# configs/llama32_qlora.yaml

model:
  name: meta-llama/Llama-3.2-3B-Instruct
  max_seq_length: 2048
  load_in_4bit: true
  dtype: bfloat16

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

data:
  train_path: data/processed/train.jsonl
  val_path:   data/processed/val.jsonl

training:
  output_dir: models/checkpoints
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 16
  num_train_epochs: 3
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  optim: adamw_8bit
  weight_decay: 0.0
  bf16: true
  fp16: false
  gradient_checkpointing: true
  logging_steps: 10
  eval_strategy: steps
  eval_steps: 50
  save_strategy: steps
  save_steps: 50
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  seed: 42
  report_to: wandb
  run_name: llama32-3b-qlora-cuad

wandb:
  project: legal-llm
  entity: null  # set to your W&B username/team
```

- [ ] **Step 8.2: Commit**

```bash
git add configs/llama32_qlora.yaml
git commit -m "feat(train): Llama-3.2-3B QLoRA training config"
```

---

## Task 9: Training script `src/train/train.py`

**Files:**
- Create: `src/train/__init__.py` (empty)
- Create: `src/train/train.py`

- [ ] **Step 9.1: Create module**

```bash
touch src/train/__init__.py
```

- [ ] **Step 9.2: Implement `src/train/train.py`**

```python
"""
Fine-tune Llama-3.2-3B with QLoRA on chunked CUAD data.

Loads config from YAML, uses Unsloth's FastLanguageModel for memory-efficient
4-bit + LoRA, wraps in TRL's SFTTrainer with response-only loss, logs to W&B.

Run: `python -m src.train.train --config configs/llama32_qlora.yaml`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def _load_jsonl_as_chat(path: Path, tokenizer) -> Dataset:
    """
    Each row becomes a chat-format example with a single user turn
    (instruction + input) and an assistant turn (output).
    """
    def to_chat(row: dict) -> dict:
        user_text = f"{row['instruction']}\n\nExcerpt:\n{row['input']}"
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": row["output"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    ds = load_dataset("json", data_files=str(path), split="train")
    return ds.map(to_chat, remove_columns=ds.column_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    import wandb
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=cfg["training"]["run_name"],
        config=cfg,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        dtype=None,  # let unsloth auto-pick
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"],
    )

    train_ds = _load_jsonl_as_chat(Path(cfg["data"]["train_path"]), tokenizer)
    val_ds = _load_jsonl_as_chat(Path(cfg["data"]["val_path"]), tokenizer)

    sft_cfg = SFTConfig(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        optim=cfg["training"]["optim"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy=cfg["training"]["eval_strategy"],
        eval_steps=cfg["training"]["eval_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        greater_is_better=cfg["training"]["greater_is_better"],
        seed=cfg["training"]["seed"],
        report_to=cfg["training"]["report_to"],
        run_name=cfg["training"]["run_name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_cfg,
    )
    trainer.train()

    best_dir = Path(cfg["training"]["output_dir"]) / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Saved best adapter to {best_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.3: Training smoke test (Phase 2 of the spec)**

Create a tiny 100-example subset and train 50 steps to confirm the whole stack works end-to-end:

```bash
head -100 data/processed/train.jsonl > data/processed/train_smoke.jsonl
cp data/processed/val.jsonl data/processed/val_smoke.jsonl  # any small set works
```

Copy `configs/llama32_qlora.yaml` → `configs/smoke.yaml`, change:
- `data.train_path: data/processed/train_smoke.jsonl`
- `data.val_path: data/processed/val_smoke.jsonl`
- `training.num_train_epochs: 1`
- `training.max_steps: 50` (add this key; SFTConfig accepts it)
- `training.eval_steps: 25`
- `training.save_steps: 50`
- `training.run_name: smoke`

Run:
```bash
python -m src.train.train --config configs/smoke.yaml
```
Expected: training completes in ~3 minutes; W&B run shows decreasing loss; `models/checkpoints/best/` exists with adapter files.

- [ ] **Step 9.4: Commit script (don't commit smoke data)**

```bash
git add src/train/train.py src/train/__init__.py
git commit -m "feat(train): QLoRA fine-tuning loop with Unsloth + TRL"
```

- [ ] **Step 9.5: Kick off full training (Phase 3 of the spec)**

```bash
python -m src.train.train --config configs/llama32_qlora.yaml
```

This takes ~4-8 hours on a 3060 depending on total chunk count. Run overnight. Monitor in W&B. On completion, `models/checkpoints/best/` contains the trained adapter.

---

## Task 10: LoRA merge script `src/train/merge.py`

**Files:**
- Create: `src/train/merge.py`

- [ ] **Step 10.1: Implement merge**

```python
"""
Merge LoRA adapter into base model and save as a standalone fp16 checkpoint.

This is what the serving Docker image ships. Merged is simpler to load at
inference than base + adapter, at the cost of ~6GB of disk.

Run: `python -m src.train.merge --adapter models/checkpoints/best --out models/merged`
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct")
    args = ap.parse_args()

    print(f"Loading base: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tok = AutoTokenizer.from_pretrained(args.base)

    print(f"Attaching adapter: {args.adapter}")
    merged = PeftModel.from_pretrained(base, str(args.adapter))
    merged = merged.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {args.out}")
    merged.save_pretrained(str(args.out), safe_serialization=True)
    tok.save_pretrained(str(args.out))
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Run merge after full training completes**

```bash
python -m src.train.merge --adapter models/checkpoints/best --out models/merged
ls -lh models/merged/
```
Expected: a `models/merged/` directory containing `model-*.safetensors`, `config.json`, tokenizer files, etc. Total ~6-7GB.

- [ ] **Step 10.3: Commit**

```bash
git add src/train/merge.py
git commit -m "feat(train): LoRA merge into standalone fp16 checkpoint"
```

---

## Task 11: Evaluation metrics in `src/eval/metrics.py`

**Files:**
- Create: `src/eval/__init__.py` (empty)
- Create: `src/eval/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 11.1: Create module**

```bash
touch src/eval/__init__.py
```

- [ ] **Step 11.2: Write failing tests**

```python
# tests/test_metrics.py
from src.common.schemas import Clause, ClauseList, ClauseType
from src.eval.metrics import (
    ClauseMetrics,
    classification_metrics,
    json_parse_rate,
    schema_valid_rate,
    span_token_f1,
)


def test_json_parse_rate():
    outputs = ['{"clauses":[]}', 'not json', '{"clauses":[]}']
    assert json_parse_rate(outputs) == 2 / 3


def test_schema_valid_rate():
    outputs = ['{"clauses":[]}', '{"not_clauses":[]}', '{"clauses":[{"type":"Governing Law","span":"x"}]}']
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
    assert m.macro_f1 == 1.0  # all types perfectly predicted-present/absent


def test_classification_metrics_false_positive_drops_precision():
    gold = [ClauseList(clauses=[])]
    pred = [ClauseList(clauses=[Clause(type=ClauseType.GOVERNING_LAW, span="x")])]
    m = classification_metrics(gold, pred)
    # precision is 0 for Governing Law (false positive), recall undefined -> 0
    gl = m.per_type[ClauseType.GOVERNING_LAW]
    assert gl.precision == 0.0


def test_span_token_f1_exact_match():
    assert span_token_f1("delaware law applies", "delaware law applies") == 1.0


def test_span_token_f1_partial():
    f1 = span_token_f1("delaware law applies here", "delaware law")
    assert 0.5 < f1 < 1.0


def test_span_token_f1_disjoint():
    assert span_token_f1("abc def", "ghi jkl") == 0.0
```

- [ ] **Step 11.3: Run — must fail**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 11.4: Implement `src/eval/metrics.py`**

```python
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
            ClauseList.model_validate_json(o)
            ok += 1
        except (json.JSONDecodeError, ValidationError):
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
        for g, p in zip(gold, pred):
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
    return ClauseMetrics(per_type=per, macro_precision=macro_p, macro_recall=macro_r, macro_f1=macro_f)


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
    max-bipartite F1 via greedy matching on token-F1."""
    per: dict[ClauseType, list[float]] = {ct: [] for ct in ClauseType}
    for g, p in zip(gold, pred):
        for ct in ClauseType:
            gs = [c.span for c in g.clauses if c.type == ct]
            ps = [c.span for c in p.clauses if c.type == ct]
            if not gs and not ps:
                continue
            if not gs or not ps:
                per[ct].append(0.0)
                continue
            # greedy match: for each gold, pick best pred; accumulate
            scores = []
            used = set()
            for gi, gs_i in enumerate(gs):
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
```

- [ ] **Step 11.5: Run — must pass**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 11.6: Commit**

```bash
git add src/eval src/eval/__init__.py src/eval/metrics.py tests/test_metrics.py
git commit -m "feat(eval): JSON, schema, classification, and span-level metrics"
```

---

## Task 12: Provider abstraction `src/eval/providers.py`

**Files:**
- Create: `src/eval/providers.py`

Uniform `generate(prompt: str) -> str` interface so `evaluate.py` doesn't know whether it's talking to a local HF model or an OpenAI API.

- [ ] **Step 12.1: Implement providers**

```python
"""
Uniform interface for generation across local HF models and remote APIs.

Each provider implements `generate(prompt: str) -> str` returning a raw
string that should ideally parse as our ClauseList JSON schema.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Provider(Protocol):
    name: str
    def generate(self, prompt: str) -> str: ...


@dataclass
class LocalHFProvider:
    name: str
    model_path: str
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
        return text


@dataclass
class OpenAIProvider:
    name: str
    model: str = "gpt-4o-mini"
    max_tokens: int = 512

    def __post_init__(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
```

- [ ] **Step 12.2: Commit**

```bash
git add src/eval/providers.py
git commit -m "feat(eval): uniform provider interface for local HF and OpenAI"
```

---

## Task 13: Single-model evaluator `src/eval/evaluate.py`

**Files:**
- Create: `src/eval/evaluate.py`

- [ ] **Step 13.1: Implement evaluator**

```python
"""
Run one provider over the test JSONL and compute all metrics.
"""
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
    per_type: dict  # clause -> {precision, recall, f1, span_f1}
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
    ap.add_argument("--provider", required=True, choices=["ft", "base-zs", "base-fs", "gpt-4o-mini"])
    ap.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prov: Provider
    few_shot = False
    if args.provider == "ft":
        prov = LocalHFProvider(name="ft", model_path="models/merged")
    elif args.provider == "base-zs":
        prov = LocalHFProvider(name="base-zs", model_path="meta-llama/Llama-3.2-3B-Instruct")
    elif args.provider == "base-fs":
        prov = LocalHFProvider(name="base-fs", model_path="meta-llama/Llama-3.2-3B-Instruct")
        few_shot = True
    else:
        prov = OpenAIProvider(name="gpt-4o-mini")
        few_shot = True

    report = evaluate(prov, args.test, few_shot=few_shot)
    args.out.write_text(json.dumps(asdict(report), indent=2))
    print(f"Saved {args.out}: macro_f1={report.macro_f1:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 13.2: Commit**

```bash
git add src/eval/evaluate.py
git commit -m "feat(eval): single-provider evaluator with full metric suite"
```

---

## Task 14: Benchmark orchestrator `src/eval/benchmark.py`

**Files:**
- Create: `src/eval/benchmark.py`

- [ ] **Step 14.1: Implement benchmark**

```python
"""
Run all 4 providers, emit eval_results.md with comparison tables.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROVIDERS = ["ft", "base-zs", "base-fs", "gpt-4o-mini"]

# USD per 1K output tokens (as of 2026-04; update if outdated).
API_COST_PER_1K_OUT = {"gpt-4o-mini": 0.0006}
# Local models: free per call (ignoring electricity).


def _run_one(provider: str) -> dict:
    out = Path(f"outputs/eval/{provider}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "python", "-m", "src.eval.evaluate",
        "--provider", provider, "--out", str(out),
    ])
    return json.loads(out.read_text())


def _cost_per_1k_contracts(provider: str, avg_out_tokens: float = 200.0) -> float:
    rate = API_COST_PER_1K_OUT.get(provider)
    if rate is None:
        return 0.0
    return rate * (avg_out_tokens / 1000) * 1000  # tokens_per_contract * contracts


def _markdown_table(reports: dict[str, dict]) -> str:
    lines = ["| Model | Macro F1 | Macro P | Macro R | JSON % | Schema % | p50 (ms) | p95 (ms) | $/1K contracts |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in PROVIDERS:
        r = reports[p]
        cost = _cost_per_1k_contracts(p)
        lines.append(
            f"| {p} | {r['macro_f1']:.3f} | {r['macro_precision']:.3f} | "
            f"{r['macro_recall']:.3f} | {r['json_parse_rate']*100:.1f}% | "
            f"{r['schema_valid_rate']*100:.1f}% | {r['p50_latency_ms']:.0f} | "
            f"{r['p95_latency_ms']:.0f} | ${cost:.4f} |"
        )
    return "\n".join(lines)


def _per_type_table(reports: dict[str, dict]) -> str:
    ft = reports["ft"]
    lines = ["| Clause | FT F1 | FT P | FT R | FT Span F1 |",
             "|---|---|---|---|---|"]
    for ct, m in sorted(ft["per_type"].items()):
        lines.append(f"| {ct} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['span_f1']:.3f} |")
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
        "- `gpt-4o-mini` = OpenAI API, 3 fixed in-context examples, same prompt.",
        "- Latencies measured on RTX 3060 6GB at batch size 1.",
        "- `$/1K contracts` assumes ~200 output tokens per contract chunk.",
    ]
    Path("eval_results.md").write_text("\n".join(md))
    print("Wrote eval_results.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 14.2: Run benchmark (requires trained model + OPENAI_API_KEY)**

```bash
export OPENAI_API_KEY=sk-...   # load from .env in real run
python -m src.eval.benchmark
cat eval_results.md
```

- [ ] **Step 14.3: Commit both the script and the results**

```bash
git add src/eval/benchmark.py eval_results.md outputs/
git commit -m "feat(eval): 4-way benchmark + committed eval_results.md"
```

---

## Task 15: Serving config + extractor wrapper

**Files:**
- Modify: `configs/serve.yaml`
- Create: `src/serve/__init__.py` (empty)
- Create: `src/serve/extractor.py`

The extractor is the single seam the API mocks in tests. Real implementation uses Outlines for constrained JSON generation.

- [ ] **Step 15.1: Populate `configs/serve.yaml`**

```yaml
model:
  path: models/merged
  max_new_tokens: 512
  timeout_seconds: 30

chunking:
  chunk_tokens: 1500
  overlap_tokens: 200

server:
  host: 0.0.0.0
  port: 8000
  cors_origins:
    - http://localhost:7860
    - http://demo:7860
```

- [ ] **Step 15.2: Create `src/serve/__init__.py`**

```bash
touch src/serve/__init__.py
```

- [ ] **Step 15.3: Implement `src/serve/extractor.py`**

```python
"""
Model-backed clause extraction. This is the one seam tests mock.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.common.prompts import build_instruction
from src.common.schemas import ClauseList, ClauseType, ExtractResponse


@dataclass
class Extractor:
    model: object  # outlines-wrapped HF model
    tokenizer: object
    generator: object  # outlines.generate.json bound to ClauseList
    model_version: str

    def extract(self, text: str, clause_types: list[ClauseType] | None) -> ExtractResponse:
        instr = build_instruction(clause_types)
        prompt = f"{instr}\n\nExcerpt:\n{text}"
        t0 = time.perf_counter()
        result: ClauseList = self.generator(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000
        return ExtractResponse(
            clauses=result.clauses,
            latency_ms=latency_ms,
            model_version=self.model_version,
        )


def build_extractor(config_path: Path = Path("configs/serve.yaml")) -> Extractor:
    """Heavy: loads model + builds Outlines generator. Call once at startup."""
    import outlines
    from outlines.models.transformers import Transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(config_path.read_text())
    model_path = cfg["model"]["path"]
    tok = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    wrapped = Transformers(hf_model, tok)
    generator = outlines.generate.json(wrapped, ClauseList)

    version = f"llama-3.2-3b-legal@{Path(model_path).name}"
    return Extractor(model=wrapped, tokenizer=tok, generator=generator, model_version=version)
```

- [ ] **Step 15.4: Commit**

```bash
git add configs/serve.yaml src/serve/__init__.py src/serve/extractor.py
git commit -m "feat(serve): extractor wrapper with Outlines constrained JSON"
```

---

## Task 16: FastAPI app `src/serve/api.py` + tests

**Files:**
- Create: `src/serve/api.py`
- Create: `tests/test_api.py`
- Modify: `tests/test_api.http`

- [ ] **Step 16.1: Write failing tests**

Create `tests/test_api.py`:

```python
from unittest.mock import MagicMock, patch

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
```

- [ ] **Step 16.2: Run — must fail**

```bash
pytest tests/test_api.py -v
```

- [ ] **Step 16.3: Implement `src/serve/api.py`**

```python
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.common.schemas import ExtractRequest, ExtractResponse
from src.serve.extractor import Extractor, build_extractor

logging.basicConfig(level=logging.INFO)
log = structlog.get_logger()

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading_extractor")
    _state["extractor"] = build_extractor(Path("configs/serve.yaml"))
    log.info("extractor_ready", version=_state["extractor"].model_version)
    yield
    _state.clear()


app = FastAPI(title="Legal LLM — Contract Clause Extractor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860", "http://demo:7860"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=rid)
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/version")
def version() -> dict:
    ex: Extractor | None = _state.get("extractor")
    return {"model": ex.model_version if ex else "not-loaded"}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    ex: Extractor | None = _state.get("extractor")
    if ex is None:
        raise HTTPException(status_code=503, detail="extractor not loaded")
    log.info("extract_request", text_len=len(req.text))
    resp = ex.extract(req.text, req.clause_types)
    log.info(
        "extract_response",
        n_clauses=len(resp.clauses),
        latency_ms=resp.latency_ms,
    )
    return resp
```

- [ ] **Step 16.4: Run — must pass**

```bash
pytest tests/test_api.py -v
```

- [ ] **Step 16.5: Populate `tests/test_api.http`**

```http
### Health
GET http://localhost:8000/healthz

### Version
GET http://localhost:8000/version

### Extract (small example)
POST http://localhost:8000/extract
Content-Type: application/json

{
  "text": "This Agreement shall be governed by the laws of the State of Delaware. Each party's total liability shall not exceed the fees paid in the prior twelve months."
}

### Extract with filtered clause types
POST http://localhost:8000/extract
Content-Type: application/json

{
  "text": "Confidential Information shall not be disclosed.",
  "clause_types": ["Confidentiality"]
}

### Extract — empty text (should 422)
POST http://localhost:8000/extract
Content-Type: application/json

{
  "text": ""
}
```

- [ ] **Step 16.6: Commit**

```bash
git add src/serve/api.py tests/test_api.py tests/test_api.http
git commit -m "feat(serve): FastAPI with Outlines extractor and mocked-extractor tests"
```

---

## Task 17: Gradio demo `src/serve/app.py` with PDF upload

**Files:**
- Create: `src/serve/app.py`

- [ ] **Step 17.1: Implement Gradio app**

```python
"""
Gradio UI: upload a PDF or paste text, get highlighted clause extraction
from the FastAPI backend.

Set API_URL env var to point at the running API (default localhost:8000).
"""
from __future__ import annotations

import json
import os
from io import BytesIO

import gradio as gr
import httpx
import pypdf
from transformers import AutoTokenizer

from src.common.chunking import chunk_text
from src.common.schemas import Clause, ClauseList, ClauseType

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "meta-llama/Llama-3.2-3B-Instruct")

_tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _pdf_to_text(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(BytesIO(file_bytes))
    return "\n\n".join(p.extract_text() or "" for p in reader.pages)


def _extract_all(full_text: str, clause_types: list[str] | None) -> list[Clause]:
    chunks = chunk_text(full_text, _tok, chunk_tokens=1500, overlap_tokens=200)
    all_clauses: list[Clause] = []
    with httpx.Client(base_url=API_URL, timeout=60.0) as client:
        for c in chunks:
            payload = {"text": c.text}
            if clause_types:
                payload["clause_types"] = clause_types
            r = client.post("/extract", json=payload)
            r.raise_for_status()
            body = r.json()
            all_clauses.extend(Clause(**c) for c in body["clauses"])
    # Dedupe by (type, span)
    seen: set[tuple[str, str]] = set()
    unique = []
    for cl in all_clauses:
        k = (cl.type.value, cl.span)
        if k not in seen:
            seen.add(k)
            unique.append(cl)
    return unique


def _render_highlighted(text: str, clauses: list[Clause]) -> list[tuple[str, str | None]]:
    """Gradio HighlightedText format: list of (text, label_or_None)."""
    if not clauses:
        return [(text, None)]
    # Sort by span occurrence in text
    spans = []
    for cl in clauses:
        idx = text.find(cl.span)
        if idx >= 0:
            spans.append((idx, idx + len(cl.span), cl.type.value))
    spans.sort()
    out = []
    cursor = 0
    for start, end, label in spans:
        if start > cursor:
            out.append((text[cursor:start], None))
        out.append((text[start:end], label))
        cursor = end
    if cursor < len(text):
        out.append((text[cursor:], None))
    return out


def extract_from_inputs(pdf_file, pasted_text: str, clause_types: list[str]) -> tuple:
    if pdf_file is not None:
        with open(pdf_file.name, "rb") as f:
            text = _pdf_to_text(f.read())
    else:
        text = pasted_text
    if not text.strip():
        return [], "{}", "Provide text or upload a PDF."
    ctypes = clause_types if clause_types else None
    clauses = _extract_all(text, ctypes)
    highlighted = _render_highlighted(text, clauses)
    table = [[c.type.value, c.span] for c in clauses]
    raw = json.dumps({"clauses": [c.model_dump() for c in clauses]}, indent=2)
    return highlighted, raw, f"Extracted {len(clauses)} clauses from {len(text)} chars."


with gr.Blocks(title="Legal LLM — Contract Clause Extractor") as demo:
    gr.Markdown("# Legal LLM — Contract Clause Extractor\n"
                "_Not legal advice. A portfolio demo._")
    with gr.Row():
        with gr.Column():
            pdf_in = gr.File(label="Contract PDF", file_types=[".pdf"])
            text_in = gr.Textbox(label="...or paste contract text", lines=10)
            types_in = gr.CheckboxGroup(
                choices=[ct.value for ct in ClauseType],
                label="Clause types (empty = all 10)",
            )
            btn = gr.Button("Extract", variant="primary")
        with gr.Column():
            status = gr.Markdown()
            highlighted = gr.HighlightedText(label="Extracted clauses in context")
            table = gr.Dataframe(headers=["Type", "Span"], label="Clauses")
            raw = gr.Code(label="Raw JSON", language="json")
    btn.click(
        extract_from_inputs,
        inputs=[pdf_in, text_in, types_in],
        outputs=[highlighted, raw, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 17.2: Manual smoke test**

Run the API and demo in two terminals:
```bash
# Terminal 1
make serve
# Terminal 2
make demo
```
Open `http://localhost:7860`, paste a short clause sample, click Extract, verify output.

- [ ] **Step 17.3: Commit**

```bash
git add src/serve/app.py
git commit -m "feat(serve): Gradio demo with PDF upload calling FastAPI backend"
```

---

## Task 18: Docker multi-stage image

**Files:**
- Modify: `docker/Dockerfile`
- Modify: `docker/docker-compose.yml`
- Create: `.dockerignore`

- [ ] **Step 18.1: Create `.dockerignore`**

```
.git/
.venv/
venv/
data/
models/checkpoints/
wandb/
outputs/
.pytest_cache/
.mypy_cache/
.ruff_cache/
__pycache__/
*.pyc
notebooks/
docs/
.github/
tests/
```

- [ ] **Step 18.2: Populate `docker/Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1.7

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# ---------- Stage 2: model-fetcher ----------
FROM python:3.11-slim AS model-fetcher
ENV PIP_NO_CACHE_DIR=1
RUN pip install huggingface_hub==0.25.2
ARG HF_MODEL_REPO=YOUR_HF_USER/legal-llm-llama32-3b
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN_VALUE=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "") && \
    HF_TOKEN="$HF_TOKEN_VALUE" python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${HF_MODEL_REPO}', local_dir='/models', local_dir_use_symlinks=False)"

# ---------- Stage 3: runtime ----------
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY --from=model-fetcher /models /models
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY src /app/src
COPY configs /app/configs
# Point serve.yaml at the baked-in model path
RUN sed -i 's|path: models/merged|path: /models|' /app/configs/serve.yaml
EXPOSE 8000
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Note on HF model repo:** Before first Docker build, push your merged model to the Hugging Face Hub:
```bash
huggingface-cli login
huggingface-cli upload YOUR_HF_USER/legal-llm-llama32-3b models/merged
```
Update `HF_MODEL_REPO` in the Dockerfile to your actual repo.

- [ ] **Step 18.3: Populate `docker/docker-compose.yml`**

```yaml
services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
      secrets:
        - HF_TOKEN
    image: legal-llm:local
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 5s
      retries: 3

  demo:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ..:/app
    environment:
      API_URL: http://api:8000
    command: >
      bash -c "pip install gradio==4.44.1 httpx==0.27.2 pypdf==5.0.1
      transformers==4.45.2 pydantic==2.9.2 &&
      python -m src.serve.app"
    ports:
      - "7860:7860"
    depends_on:
      api:
        condition: service_healthy

secrets:
  HF_TOKEN:
    environment: HF_TOKEN
```

- [ ] **Step 18.4: Build image locally**

```bash
export HF_TOKEN=hf_...   # load from .env
docker build -f docker/Dockerfile -t legal-llm:local --secret id=HF_TOKEN,env=HF_TOKEN .
docker run --rm --gpus all -p 8000:8000 legal-llm:local &
sleep 30
curl http://localhost:8000/healthz
docker ps -q | xargs -r docker kill
```

- [ ] **Step 18.5: Commit**

```bash
git add docker/ .dockerignore
git commit -m "feat(docker): multi-stage image with baked-in model"
```

---

## Task 19: GitHub Actions CI `ci.yml`

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 19.1: Implement CI workflow**

```yaml
name: CI
on:
  push:
    branches: ["**"]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements-dev.txt
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/
      # mypy against a narrow subset — deep ML deps are too heavy for CI
      - run: pip install pydantic fastapi httpx structlog pyyaml
      - run: mypy src/common src/serve

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: |
          pip install -r requirements-dev.txt
          pip install pydantic==2.9.2 fastapi==0.115.0 httpx==0.27.2 \
            structlog==24.4.0 pyyaml==6.0.2 transformers==4.45.2
      - run: pytest tests/test_schemas.py tests/test_metrics.py \
              tests/test_validate.py tests/test_api.py -v

  docker-build-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Build builder stage only (skip HF-gated model fetch)
        run: |
          docker build -f docker/Dockerfile --target builder -t legal-llm:builder .
```

- [ ] **Step 19.2: Populate `.github/workflows/eval.yml` (release workflow)**

```yaml
name: Release Docker Image
on:
  push:
    tags: ["v*"]
  workflow_dispatch:

permissions:
  contents: read
  packages: write

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Prepare secret file
        run: echo "${{ secrets.HF_TOKEN }}" > hf_token.txt
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/legal-llm:${{ github.ref_name }}
            ghcr.io/${{ github.repository_owner }}/legal-llm:latest
          secret-files: |
            HF_TOKEN=hf_token.txt
      - run: rm hf_token.txt
```

- [ ] **Step 19.3: Commit**

```bash
git add .github/workflows/
git commit -m "ci: lint + test + docker build-check; release on tag → ghcr.io"
```

---

## Task 20: README polish

**Files:**
- Modify: `README.md`

- [ ] **Step 20.1: Rewrite README with full content**

Replace `README.md`:

````markdown
# Legal LLM — Contract Clause Extractor

A QLoRA fine-tune of **Llama-3.2-3B** on the CUAD contract dataset, extracting 10 high-value commercial clause types as structured JSON. Shipped as a Dockerized FastAPI service with a Gradio PDF-upload demo.

**Disclaimer:** This is a portfolio project. Not legal advice.

## Overview

Given a commercial-contract excerpt, the model returns:

```json
{
  "clauses": [
    {"type": "Governing Law", "span": "This Agreement shall be governed by..."},
    {"type": "Liability Cap", "span": "Total liability shall not exceed..."}
  ]
}
```

**Target clauses (10):** Governing Law, Indemnification, Non-Compete, Termination for Convenience, Liability Cap, Exclusivity, IP Assignment, Confidentiality, Change of Control, Auto-Renewal.

**Live demo (HF Spaces):** https://huggingface.co/spaces/YOUR_USER/legal-llm
**W&B runs:** https://wandb.ai/YOUR_USER/legal-llm
**Docker image:** `ghcr.io/YOUR_USER/legal-llm:latest`

## Architecture

```
CUAD → chunker → JSONL → Unsloth + TRL QLoRA (Llama-3.2-3B)
                                    │
                            merged fp16 checkpoint
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
   FastAPI (/extract)       Gradio UI (PDF upload)    GPT-4o-mini baseline
   Outlines-constrained     calls API over HTTP       (eval comparison)
   JSON output
```

See [`docs/superpowers/specs/2026-04-21-legal-llm-contract-extractor-design.md`](docs/superpowers/specs/2026-04-21-legal-llm-contract-extractor-design.md) for the full design.

## Eval Results

See [`eval_results.md`](eval_results.md) for the full 4-way comparison. Headline:

- Fine-tuned 3B beats base-model few-shot by +X macro-F1
- Within Y macro-F1 points of GPT-4o-mini at ~1/Z the cost per 1K contracts

## Setup

```bash
git clone <repo>
cd legal-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

Requires CUDA 12.1 + PyTorch 2.4. Tested on RTX 3060 6GB.

## Training

```bash
# 1. Prepare CUAD data
make data

# 2. Explore
jupyter notebook notebooks/01_data_exploration.ipynb

# 3. Train (4-8 hours on 3060)
export WANDB_API_KEY=...
make train

# 4. Merge LoRA
make merge

# 5. Evaluate all 4 models
export OPENAI_API_KEY=...
make eval
cat eval_results.md
```

## Demo

**Via Docker (recommended — model baked in):**
```bash
docker pull ghcr.io/YOUR_USER/legal-llm:latest
docker run --gpus all -p 8000:8000 ghcr.io/YOUR_USER/legal-llm:latest
# then, in another terminal:
API_URL=http://localhost:8000 python -m src.serve.app
```

**From source:**
```bash
make serve    # terminal 1: FastAPI on :8000
make demo     # terminal 2: Gradio on :7860
```

Open `http://localhost:7860`, upload a PDF, see clauses extracted.

## Limitations

- **English only.** CUAD is English; the model hasn't seen other languages.
- **3060 6GB constrains context** to ~1500 training tokens per chunk. Very long unbroken paragraphs may straddle chunk boundaries.
- **10 clause types only.** Not a general legal analyzer. CUAD has 41 types; we chose 10 for depth over breadth.
- **No authentication.** The API is open; add API-key auth before any real deployment.
- **Outlines constrained decoding** can occasionally time out on unusual inputs. Falls back to unconstrained + JSON repair.
- **Test set is ~50 contracts.** F1 confidence intervals are wide; see `eval_results.md` for bootstrap CIs.
- **Not legal advice.** Obviously.

## License

MIT (code). CUAD dataset is CC BY 4.0.
````

- [ ] **Step 20.2: Commit**

```bash
git add README.md
git commit -m "docs: full README with setup, training, demo, limitations"
```

---

## Task 21: HF Spaces deployment

**Files:**
- Create: `spaces/README.md` (separate repo for HF Spaces)

- [ ] **Step 21.1: Create a separate HF Space repo** (done through the HF web UI). Type: Gradio. SDK version: 4.44.1.

- [ ] **Step 21.2: Push Gradio app + its minimal deps to Space**

The Space's `app.py` points `API_URL` at your HF Spaces-hosted API (if you choose to host the API on Spaces too) or at a public endpoint you host elsewhere. Free Spaces CPU is enough for Gradio, but the API will be slow — README is explicit about this.

Alternative: Run only the Gradio app on Spaces with `API_URL=https://your-domain.com:8000` pointing at your home server (via a tunnel like `cloudflared`). Document this setup in `docs/hosting.md`.

- [ ] **Step 21.3: Update main README with Spaces URL** once live.

---

## Task 22: Final integration walk-through

- [ ] **Step 22.1: Fresh clone test**

On a different machine (or fresh `git clone` into `/tmp`):
```bash
cd /tmp
git clone <your-repo> legal-llm-verify
cd legal-llm-verify
docker pull ghcr.io/YOUR_USER/legal-llm:latest
docker run --rm --gpus all -p 8000:8000 ghcr.io/YOUR_USER/legal-llm:latest &
sleep 60
curl -X POST http://localhost:8000/extract \
  -H 'content-type: application/json' \
  -d '{"text":"This Agreement shall be governed by Delaware law."}'
```
Expected: returns valid JSON with `{"clauses":[{"type":"Governing Law",...}], ...}`.

- [ ] **Step 22.2: Tag and release**

```bash
git tag v0.1.0
git push origin main --tags
```
Watch GitHub Actions build and publish the image.

- [ ] **Step 22.3: Link all portfolio artifacts in README** — the W&B project URL, HF Space URL, `docker pull` command, `eval_results.md` link.

- [ ] **Step 22.4: Final commit**

```bash
git add README.md
git commit -m "docs: link portfolio artifacts (W&B, Spaces, ghcr.io image)"
git push
```

---

## Self-Review Checklist (for the engineer before merging)

- [ ] All 22 tasks complete; `git log` shows granular commits.
- [ ] `make ci` passes locally (lint + tests).
- [ ] `make eval` populates `eval_results.md` with all 4 rows.
- [ ] Public W&B project exists and has at least one run with decreasing loss.
- [ ] `docker pull ghcr.io/YOUR_USER/legal-llm:latest && docker run` works on a fresh machine.
- [ ] HF Space loads; PDF upload extracts clauses end-to-end.
- [ ] README has zero placeholders (`YOUR_USER`, `YOUR_HF_USER`, `TBD`) — replaced with your actual handles.
- [ ] Limitations section honest about test-set size, VRAM constraints, Outlines brittleness.

---

## Execution Notes

- **Phases 1-4 (Tasks 0-7)** are CPU-only except for the full CUAD load. Run end-to-end before touching training.
- **Phase 5 (Tasks 8-9)** requires GPU; Task 9 Step 9.3 is the "smoke test" from the spec — don't skip it.
- **Phase 6 (Task 9 Step 9.5 + Tasks 10-14)** is the overnight training + eval loop. Plan for 1-2 iterations: if eval is disappointing, tweak the config and re-run.
- **Phase 7 (Tasks 15-20)** is pure engineering — no more GPU required (except for serving smoke tests).
- **Task 21 (HF Spaces)** can be done any time after Task 20.
