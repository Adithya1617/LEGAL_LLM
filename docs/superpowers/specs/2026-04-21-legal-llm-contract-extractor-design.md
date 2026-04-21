# Legal LLM — Contract Clause Extractor (Design Spec)

**Date:** 2026-04-21
**Status:** Approved for implementation planning
**Author:** adi_vai_vis@aiplanet.com
**Hardware:** Pop!_OS, RTX 3060 6GB VRAM
**Goal:** Production-grade portfolio project: fine-tune a small open-weights LLM for a real-world specialized task, ship as a Dockerized FastAPI service with a public demo.

---

## 1. Problem Statement

Build a specialized 3B-parameter model that, given an excerpt of a commercial contract, returns structured JSON identifying which of 10 high-value clause types are present and extracting their exact spans from the text. Ship it as a Dockerized FastAPI service with a Gradio demo that accepts PDF uploads.

### Target clause types (10, from CUAD)
Governing Law, Indemnification, Non-Compete, Termination for Convenience, Liability Cap, Exclusivity, IP Assignment, Confidentiality, Change of Control, Auto-Renewal.

Final list may shift slightly after exploring CUAD label counts in the data-exploration notebook; the commitment is "10 high-value clauses," not this exact list.

---

## 2. Success Criteria

1. **Model quality:** Fine-tuned Llama-3.2-3B beats base-model few-shot by ≥ +10 macro-F1 on the held-out test set. Target: within 5 macro-F1 points of GPT-4o-mini few-shot.
2. **JSON reliability:** ≥ 99% of production outputs parse as valid JSON matching the pydantic schema. Enforced via constrained decoding (Outlines).
3. **Latency:** p95 inference latency < 2s per chunk on the RTX 3060 (batch size 1).
4. **Reproducibility:** Fresh clone + `make docker-build` + `docker run` produces a working API whose outputs match the committed eval numbers.
5. **Portfolio artifacts:**
   - Public W&B project URL linked in README
   - Public Gradio demo on Hugging Face Spaces
   - `docker pull ghcr.io/<user>/legal-llm` image in README
   - Committed `eval_results.md` with full comparison tables

---

## 3. Key Design Decisions (with rationale)

| Decision | Choice | Why |
|---|---|---|
| Task framing | Generative, all-clauses-at-once, top-10 only | Real product behavior (one call per chunk); manageable output length for 6GB; "I picked the 10 highest-value clauses" shows judgment |
| Model | Llama-3.2-3B-Instruct | Recognizable name, fits 6GB with standard QLoRA tricks, no heroic memory management needed |
| Fine-tuning method | QLoRA (4-bit NF4 base + LoRA adapters) | Standard for single-consumer-GPU fine-tuning; ~1.3% trainable params |
| Training framework | Unsloth + TRL `SFTTrainer` | ~2× speed, ~50% VRAM savings vs vanilla HF; makes 3B on 6GB comfortable |
| Train/test split | By contract, not by chunk | Only honest way to evaluate; prevents train/test leakage across paragraphs of the same doc |
| JSON reliability | Constrained decoding via Outlines | "Mostly JSON" is not production; schema-guided generation eliminates parse failures |
| Baselines | Fine-tuned vs base Llama-3.2-3B zero-shot vs base few-shot vs GPT-4o-mini few-shot | Proves fine-tuning helped AND shows small specialist vs large generalist narrative |
| Additional eval | Latency + $/1K contracts table | Makes "production grade" concrete; shows you think about cost |
| Experiment tracking | Weights & Biases, public project URL | Industry standard; public URL is a portfolio artifact |
| Serving | FastAPI backend + Gradio frontend (separate processes, HTTP) | Clean API/UI separation; Gradio replaceable without touching model |
| PDF ingestion | pypdf in the Gradio layer, not the API | API stays pure (text in, JSON out); PDF is a UX concern |
| CI | Ruff + pytest + API contract tests with mocked model + Docker build on tag | Industry-floor CI; eval itself is run manually on release branches |
| Docker | Multi-stage, model baked into image | Portfolio-optimized: one slow pull, instant `docker run` forever after |
| Explicitly out of scope | Multi-language, RLHF/DPO, GGUF export, auth, persistence | YAGNI; portfolio scope |

---

## 4. Architecture

```
                         ┌──────────────────────────┐
                         │  CUAD dataset (raw)      │
                         └──────────┬───────────────┘
                                    │  src/data/prepare.py
                                    ▼
                         ┌──────────────────────────┐
                         │  Processed JSONL         │
                         │  (instruction, input,    │
                         │   output) + pydantic     │
                         │   validation             │
                         └──────────┬───────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ src/train/       │  │ src/eval/        │  │ Test split       │
    │ train.py         │  │ evaluate.py +    │  │ (contract-level  │
    │ Unsloth+TRL      │  │ benchmark.py     │  │ holdout)         │
    │ QLoRA on         │  │ FT vs base-zs vs │  │                  │
    │ Llama-3.2-3B     │  │ base-fs vs       │  │                  │
    │                  │  │ GPT-4o-mini +    │  │                  │
    │                  │  │ latency + cost   │  │                  │
    └────────┬─────────┘  └────────┬─────────┘  └──────────────────┘
             │                     │
             │                     ▼
             │           ┌──────────────────┐
             │           │ W&B + committed  │
             │           │ eval_results.md  │
             │           └──────────────────┘
             ▼
    ┌──────────────────┐  src/train/merge.py
    │ LoRA adapter     │ ───────────────────────► merged fp16 model
    └────────┬─────────┘                          (models/merged/)
             │
             ▼
    ┌──────────────────┐      ┌──────────────────┐     ┌──────────────────┐
    │ src/serve/api.py │◄─────│ src/serve/app.py │◄────│ User: uploads    │
    │ FastAPI          │ HTTP │ Gradio + pypdf   │     │ PDF in browser   │
    │ POST /extract    │      │ (PDF→chunks→API) │     │                  │
    │ Outlines JSON-   │      │                  │     │                  │
    │ constrained      │      │                  │     │                  │
    └──────────────────┘      └──────────────────┘     └──────────────────┘
             ▲
             │
    ┌──────────────────┐
    │ Docker image     │
    │ (model baked in) │ ──► ghcr.io/<user>/legal-llm
    └──────────────────┘
```

---

## 5. Components (detailed)

### 5.1 Data pipeline — `src/data/`

**Source:** CUAD v1 from Hugging Face (`theatticusproject/cuad-qa`). 510 contracts, 41 clause types, originally SQuAD-style extractive QA.

**`prepare.py` — transforms CUAD into training JSONL:**
1. Download CUAD via `datasets.load_dataset`.
2. Filter to the 10 target clause types.
3. Chunk each contract to ~1500 tokens with 200-token overlap (Llama-3.2 tokenizer). Max training seq length is 2048; we leave headroom for the instruction and output.
4. For each chunk, aggregate all gold spans falling within it:
   ```json
   {
     "instruction": "Extract the following clause types from this contract excerpt. Return a JSON object with a 'clauses' array. Clause types: [Governing Law, Indemnification, Non-Compete, Termination for Convenience, Liability Cap, Exclusivity, IP Assignment, Confidentiality, Change of Control, Auto-Renewal]",
     "input": "<contract chunk text>",
     "output": "{\"clauses\": [{\"type\": \"Governing Law\", \"span\": \"This Agreement shall be governed by the laws of Delaware.\"}]}"
   }
   ```
5. **Negative examples:** ensure ~30% of chunks have `{"clauses": []}`. Prevents the model from learning to always find something.
6. **Split by contract** (80/10/10 train/val/test), stratified by contract length bucket so each split has long and short documents.
7. Write to `data/processed/{train,val,test}.jsonl`.

**`validate.py` — pydantic schema enforcement:**
- Schema:
  ```python
  class Clause(BaseModel):
      type: Literal["Governing Law", "Indemnification", ...]  # 10 types
      span: str

  class ClauseList(BaseModel):
      clauses: list[Clause]
  ```
- Every training row's `output` must parse against `ClauseList`.
- Every `span` must be a substring of the corresponding `input` (after whitespace normalization). Fail the prepare step if not.
- Emit counts per clause type per split. Warn if any clause type has < 20 train examples.

### 5.2 Training — `src/train/`

**Model & quantization:**
- Base: `meta-llama/Llama-3.2-3B-Instruct`
- 4-bit NF4 via bitsandbytes, loaded through `unsloth.FastLanguageModel.from_pretrained(..., load_in_4bit=True)`
- Max sequence length: 2048

**LoRA configuration (starting point):**
- Rank: 16
- Alpha: 32
- Dropout: 0.05
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Trainable params: ~40M of 3B (~1.3%)

**Training hyperparameters (`configs/llama32_qlora.yaml` — renamed from `phi3_qlora.yaml`):**
- Per-device batch size: 1
- Gradient accumulation: 16 (effective batch = 16)
- LR: 2e-4, cosine schedule, 5% warmup
- Epochs: 3 (tune via val loss)
- Gradient checkpointing: on
- Optimizer: `adamw_8bit`
- Precision: bf16
- Eval every 50 steps on val split; save best by val loss

**Loss masking:** response-only loss via TRL's `DataCollatorForCompletionOnlyLM`. Instruction tokens don't contribute to gradient.

**W&B:** `wandb.init(project="legal-llm")` at top of `train.py`. SFTTrainer auto-logs loss / lr / grad norm / GPU memory. Custom callback logs 3 sample generations on val every N steps for qualitative inspection.

**`merge.py`:** after training, merge LoRA weights into base model via `model.merge_and_unload()`, save fp16 to `models/merged/`. This is what the Docker image ships.

### 5.3 Evaluation — `src/eval/`

**`evaluate.py` — single-model metrics over test set:**

- **Structural metrics:**
  - JSON parse rate
  - Pydantic schema validation rate
- **Classification metrics (per clause type + macro):**
  - Precision / Recall / F1 on "is clause present in this chunk?"
- **Span-level metrics (per clause type + macro):**
  - Exact match rate
  - Token-level F1 between predicted and gold span
  - When multiple gold spans exist per clause in a chunk, compute set-level F1 with span-matching via token overlap

**`benchmark.py` — orchestrates comparisons:**

Runs `evaluate.py` with four model configurations:
1. Fine-tuned Llama-3.2-3B (merged) on the 3060
2. Base Llama-3.2-3B zero-shot on the 3060
3. Base Llama-3.2-3B with 3 in-context examples (same 3 examples for every test item) on the 3060
4. GPT-4o-mini with same 3 in-context examples via OpenAI API

Measurements per model:
- All metrics above
- Latency p50 / p95 from `time.perf_counter` around the generation call, batch 1
- $ per 1K contracts (API models use published pricing; local models: $0)

Outputs:
- `eval_results.md` committed to repo — markdown table + bootstrap confidence intervals
- W&B run artifact with same data

### 5.4 Serving — `src/serve/`

**`api.py` — FastAPI (the product):**

- `POST /extract`
  - Request: `{ "text": str, "clause_types": list[str] | null }`
  - Response: `{ "clauses": [{ "type": str, "span": str }], "latency_ms": float, "model_version": str }`
- `GET /healthz` — returns 200 after model is loaded
- `GET /version` — returns `{ "model": "llama-3.2-3b-legal", "lora_rev": "...", "build_sha": "..." }`

**Implementation details:**
- **Constrained decoding:** Outlines' `generate.json(model, ClauseList)` guarantees schema-valid output. Hard timeout (configurable, default 30s) with fallback to unconstrained generation + JSON-repair.
- **Model lifecycle:** loaded once at startup via FastAPI `lifespan` context; uses Unsloth's `FastLanguageModel.for_inference()` for 2× generation speed.
- **Logging:** `structlog` JSON logs: request ID, latency, input/output token counts, errors.
- **CORS:** configured for the Gradio demo origin.
- **Config:** `configs/serve.yaml` — model path, max concurrent requests, chunk size, timeout.

**`app.py` — Gradio (the demo):**

- Inputs: PDF upload OR text paste
- Pipeline: `pypdf` extracts text → chunker splits into ~1500-token chunks (same logic as training) → call `/extract` per chunk → merge results → dedupe by span text
- Outputs:
  - Table of extracted clauses (type, span, source chunk index)
  - Original text with highlighted spans (color-coded by clause type)
  - Raw JSON response (collapsible)
- Backend URL via env var: `API_URL` (default `http://localhost:8000`)

### 5.5 Testing — `tests/`

- **`test_data.py`** — pydantic schema validation; `prepare.py` idempotency (running twice produces identical output); split integrity (no contract appears in two splits); negative-example ratio sanity check (25–35%).
- **`test_api.py`** — `httpx.TestClient` against FastAPI with the model **mocked** (monkeypatch the extractor function to return canned responses). Test cases: happy path, empty text, oversized text (>max context), unknown clause types, malformed request, healthz, version. Target: ~20 tests, all CPU-fast.
- **`test_api.http`** — hand-runnable real API calls in the `REST Client` VS Code extension format for manual smoke tests against a running server.

### 5.6 Infrastructure — `docker/` and `.github/workflows/`

**`docker/Dockerfile` — multi-stage:**

1. **Stage `builder`:** `python:3.11-slim`. Install `uv`, install pinned `requirements.txt` into `/opt/venv`.
2. **Stage `model-fetcher`:** `python:3.11-slim`. `huggingface_hub.snapshot_download` of the published fine-tuned model into `/models`. Uses `--mount=type=secret,id=HF_TOKEN` during build for private-repo support.
3. **Stage `runtime`:** `nvidia/cuda:12.1-cudnn9-runtime-ubuntu22.04`. Copy venv from `builder`, model from `model-fetcher`, source from context. Entry: `uvicorn src.serve.api:app --host 0.0.0.0 --port 8000`.

Expected image size: ~6.5 GB (CUDA runtime ~2GB + venv ~1.5GB + model ~2.5GB + OS ~500MB).

**`docker/docker-compose.yml` — two services:**

- `api` — builds from Dockerfile, exposes 8000, requires nvidia runtime (`deploy.resources.reservations.devices` with GPU capability)
- `demo` — lightweight image running Gradio, `depends_on: api`, `API_URL=http://api:8000`, exposes 7860. CPU-only.

**`.github/workflows/ci.yml` (new):**

- Triggers: `push` to any branch, `pull_request`
- Jobs:
  1. `lint` — `ruff check`, `ruff format --check`, `mypy src/`
  2. `test` — `pytest tests/` (model mocked, CPU-only, ~1 min)
  3. `build` — `docker build` through `builder` stage only (verifies deps install cleanly without HF auth)

**`.github/workflows/eval.yml` (existing, to be populated):**

- Triggers: `workflow_dispatch` (manual) + push of tag matching `v*`
- Job: build full Docker image, publish to `ghcr.io/<user>/legal-llm:${TAG}` and `:latest`
- Uses `docker/build-push-action@v5` with GHCR credentials from `GITHUB_TOKEN`
- Runs on `ubuntu-latest`; GPU is a runtime concern, not a build concern

**GitHub secrets required:** `HF_TOKEN` (build-time model pull).

### 5.7 Repository hygiene

- `requirements.txt` fully pinned (already generated from venv, 164 packages)
- `requirements-dev.txt` separated (ruff, mypy, pytest) so serving image stays lean
- Pre-commit hooks: `ruff`, `ruff format`, `check-yaml`, `check-added-large-files`
- Makefile targets:
  ```
  data:          python -m src.data.prepare && python -m src.data.validate
  train:         python -m src.train.train --config configs/llama32_qlora.yaml
  eval:          python -m src.eval.benchmark
  serve:         uvicorn src.serve.api:app --reload
  docker-build:  docker build -f docker/Dockerfile -t legal-llm:local .
  ```

---

## 6. Build Order

Seven phases, each ending in something commit-able and independently demo-able.

| Phase | Deliverable | Validation |
|---|---|---|
| 1. Data pipeline | CUAD → processed JSONL + pydantic validation + exploration notebook with counts | `pytest tests/test_data.py` green; label counts sensible |
| 2. Training smoke test | 50 steps on 100 examples; loss decreases; generates plausible output | Visual check of sample generations |
| 3. Full training | 3-epoch run; W&B logged; merged checkpoint | Val-loss curve sane; public W&B URL exists |
| 4. Eval harness | `evaluate.py` + `benchmark.py`; `eval_results.md` committed with 4 baselines | Table has FT + base-zs + base-fs + GPT-4o-mini rows + latency column |
| 5. FastAPI | `/extract` with Outlines constrained JSON; tests pass | `pytest tests/test_api.py` green; manual `.http` calls work |
| 6. Gradio + PDF | `app.py` calls API; PDF upload end-to-end | Upload a random NDA; see clauses highlighted |
| 7. Docker + CI | Image on `ghcr.io`; CI green on PRs; README polished | `docker pull && docker run` on a clean machine works |

---

## 7. Risks & Limitations

Documented in README and addressed honestly:

1. **6GB VRAM ceiling.** If evaluation shows underfitting, we can't just scale LoRA larger without reducing seq length or batch size. Mitigation: eval early and often in Phase 2; fallback to Llama-3.2-1B if stuck.
2. **CUAD class imbalance.** Some of the 10 target clauses have 100+ training examples, others ~30. Low-count-clause F1 will be noisy. Mitigation: report per-clause F1 (don't hide behind macro-avg); optionally oversample rare clauses.
3. **Small test set.** Contract-level holdout gives ~50 test contracts. F1 confidence intervals will be wide. Mitigation: compute bootstrap CIs in `benchmark.py` and report them.
4. **Constrained decoding brittleness.** Outlines enforces schema but can loop or hit length caps if confused. Mitigation: hard timeout + fallback to unconstrained + JSON repair step.
5. **HF Spaces free tier is CPU.** Hosted Gradio demo will be slow (~5-10s per contract chunk). README explicit: zero-friction demo is Spaces; fast demo is local Docker run.
6. **Not legal advice.** Prominent disclaimer in README and Gradio UI.

---

## 8. Out of Scope (YAGNI)

- Multi-language support (English contracts only)
- Long-context handling beyond chunking (no sliding-window attention; no RAG)
- Continued fine-tuning, DPO, or RLHF (SFT only)
- Quantized export formats (GGUF, AWQ) — merged fp16 + Outlines is enough
- API authentication (noted as "add API key auth before deploying anywhere real")
- Database / persistence (stateless API)

---

## 9. Project Structure (already scaffolded)

```
src/data/{prepare,validate}.py              # Phase 1
src/train/{train,merge}.py                  # Phases 2, 3
src/eval/{evaluate,benchmark}.py            # Phase 4
src/serve/{api,app}.py                      # Phases 5, 6
configs/llama32_qlora.yaml                  # (rename from phi3_qlora.yaml)
configs/serve.yaml
notebooks/01_data_exploration.ipynb         # Phase 1 exploration
tests/{test_data,test_api}.py
tests/test_api.http
docker/{Dockerfile,docker-compose.yml}      # Phase 7
.github/workflows/{ci,eval}.yml             # Phase 7 (ci.yml to be added)
Makefile                                    # fill in all targets during Phase 7
requirements.txt                            # pinned (164 packages from venv)
README.md                                   # polish in Phase 7
```
