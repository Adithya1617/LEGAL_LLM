"""Model-backed clause extraction. This is the one seam tests mock."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.common.prompts import build_instruction
from src.common.schemas import ClauseList, ClauseType, ExtractResponse


@dataclass
class Extractor:
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
    return Extractor(generator=generator, model_version=version)
