"""Gradio UI: upload a PDF or paste text, get highlighted clause extraction
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
from src.common.schemas import Clause, ClauseType

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "unsloth/Llama-3.2-3B-Instruct")

try:
    _tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
except Exception:
    _tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")


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
            all_clauses.extend(Clause(**cl) for cl in body["clauses"])
    seen: set[tuple[str, str]] = set()
    unique: list[Clause] = []
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
    spans = []
    for cl in clauses:
        idx = text.find(cl.span)
        if idx >= 0:
            spans.append((idx, idx + len(cl.span), cl.type.value))
    spans.sort()
    out: list[tuple[str, str | None]] = []
    cursor = 0
    for start, end, label in spans:
        if start > cursor:
            out.append((text[cursor:start], None))
        out.append((text[start:end], label))
        cursor = end
    if cursor < len(text):
        out.append((text[cursor:], None))
    return out


def extract_from_inputs(
    pdf_file, pasted_text: str, clause_types: list[str]
) -> tuple[list, str, str]:
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
    raw = json.dumps({"clauses": [c.model_dump() for c in clauses]}, indent=2)
    return highlighted, raw, f"Extracted {len(clauses)} clauses from {len(text)} chars."


with gr.Blocks(title="Legal LLM — Contract Clause Extractor") as demo:
    gr.Markdown("# Legal LLM — Contract Clause Extractor\n_Not legal advice. A portfolio demo._")
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
            raw = gr.Code(label="Raw JSON", language="json")
    btn.click(
        extract_from_inputs,
        inputs=[pdf_in, text_in, types_in],
        outputs=[highlighted, raw, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
