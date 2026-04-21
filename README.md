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

See [`docs/superpowers/specs/2026-04-21-legal-llm-contract-extractor-design.md`](docs/superpowers/specs/2026-04-21-legal-llm-contract-extractor-design.md) for the full design and [`docs/superpowers/plans/2026-04-21-legal-llm-contract-extractor.md`](docs/superpowers/plans/2026-04-21-legal-llm-contract-extractor.md) for the implementation plan.

## Eval Results

See [`eval_results.md`](eval_results.md) for the full 4-way comparison. Headline:

- Fine-tuned 3B vs base-model few-shot: +X macro-F1
- Fine-tuned 3B vs GPT-4o-mini few-shot: -Y macro-F1 at ~1/Z the cost per 1K contracts

## Setup

Prerequisites: Python 3.12, CUDA 12.4, PyTorch 2.6+, RTX 3060 6GB or better.

```bash
git clone <repo>
cd legal-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

## Training

```bash
# 1. Prepare CUAD data
make data

# 2. Explore / verify label mapping
jupyter notebook notebooks/01_data_exploration.ipynb

# 3. Train (4-8 hours on 3060)
export WANDB_API_KEY=...
make train

# 4. Merge LoRA into standalone fp16
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
# in another terminal:
API_URL=http://localhost:8000 python -m src.serve.app
```

**From source:**

```bash
make serve    # terminal 1: FastAPI on :8000
make demo     # terminal 2: Gradio on :7860
```

Open `http://localhost:7860`, upload a PDF, see clauses extracted.

## Limitations

- **English only.** CUAD is English; the model has not seen other languages.
- **3060 6GB constrains context** to ~1500 training tokens per chunk. Long unbroken paragraphs may straddle chunk boundaries.
- **10 clause types only.** Not a general legal analyzer. CUAD has 41 types; we chose 10 for depth over breadth.
- **No authentication.** API is open; add API-key auth before any real deployment.
- **Outlines constrained decoding** can occasionally time out on unusual inputs. Falls back to unconstrained + JSON repair.
- **Test set is ~50 contracts.** F1 confidence intervals are wide; `eval_results.md` reports bootstrap CIs.
- **Not legal advice.** Obviously.

## License

MIT (code). CUAD dataset is CC BY 4.0.
