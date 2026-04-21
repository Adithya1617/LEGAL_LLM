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
        chunks.append(
            Chunk(text=text[start_char:end_char], start_char=start_char, end_char=end_char)
        )
        if end == len(input_ids):
            break
        i += stride
    return chunks
