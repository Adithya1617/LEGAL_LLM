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
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from src.common.chunking import chunk_text
from src.common.prompts import build_instruction
from src.common.schemas import ClauseList, ClauseType

# --- Mapping from CUAD question labels to our 10 clause types ----------------
# CUAD question labels are long and specific. We canonicalize to ClauseType.
# NOTE: This mapping is a starting point — the notebook `01_data_exploration.ipynb`
# (Task 7) dumps exact CUAD category strings and may refine this mapping.
CUAD_TO_CLAUSE_TYPE: dict[str, ClauseType] = {
    "Governing Law": ClauseType.GOVERNING_LAW,
    "Indemnification": ClauseType.INDEMNIFICATION,
    "Non-Compete": ClauseType.NON_COMPETE,
    "Termination For Convenience": ClauseType.TERMINATION_FOR_CONVENIENCE,
    "Cap On Liability": ClauseType.LIABILITY_CAP,
    "Exclusivity": ClauseType.EXCLUSIVITY,
    "Ip Ownership Assignment": ClauseType.IP_ASSIGNMENT,
    # Confidentiality: placeholder — refine in notebook if CUAD uses a different label
    "Non-Disparagement": ClauseType.CONFIDENTIALITY,
    "Change Of Control": ClauseType.CHANGE_OF_CONTROL,
    "Renewal Term": ClauseType.AUTO_RENEWAL,
}

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
FALLBACK_MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"


def aggregate_gold_spans_in_chunk(
    gold: list[dict], chunk_start: int, chunk_end: int, chunk_content: str
) -> list[dict]:
    """Keep gold spans whose char range falls inside [chunk_start, chunk_end]
    AND whose span text appears verbatim in chunk_content."""
    kept: list[dict] = []
    for g in gold:
        if (
            g["start_char"] >= chunk_start
            and g["end_char"] <= chunk_end
            and g["span"] in chunk_content
        ):
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


def _load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception:
        return AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)


def _cuad_to_contracts() -> dict[str, dict]:
    """
    Load CUAD-v1 and regroup by contract.

    Returns: {"gold": {contract_id: [{type, span, start_char, end_char}, ...]},
              "text": {contract_id: full_text}}
    """
    ds = load_dataset("theatticusproject/cuad-qa", split="train")
    gold_by_contract: dict[str, list[dict]] = {}
    text_by_contract: dict[str, str] = {}
    for row in ds:
        cid = row["id"].split("_")[0]
        text_by_contract[cid] = row["context"]
        category = row["question"].split('"')[1] if '"' in row["question"] else row["question"]
        ctype = CUAD_TO_CLAUSE_TYPE.get(category)
        if ctype is None:
            continue
        gold_by_contract.setdefault(cid, [])
        for start, text in zip(
            row["answers"]["answer_start"], row["answers"]["text"], strict=False
        ):
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
    clause_list = ClauseList(clauses=[{"type": g["type"], "span": g["span"]} for g in kept])
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
    tok = _load_tokenizer()
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
                    kept = aggregate_gold_spans_in_chunk(gold, c.start_char, c.end_char, c.text)
                    if not kept and split_name == "train" and rng.random() > negative_keep_prob:
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
