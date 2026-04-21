"""Uniform interface for generation across local HF models and remote APIs.

Each provider implements `generate(prompt: str) -> str` returning a raw string
that should parse as our ClauseList JSON schema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
        return self.tokenizer.decode(out[0, inputs.shape[1] :], skip_special_tokens=True)


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
