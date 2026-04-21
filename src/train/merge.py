"""Merge LoRA adapter into base model and save as standalone fp16 checkpoint.

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
