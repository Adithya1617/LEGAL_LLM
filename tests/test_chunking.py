from transformers import AutoTokenizer

from src.common.chunking import chunk_text

TOKENIZER_NAME = "unsloth/Llama-3.2-3B-Instruct"


def _tok():
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception:
        return AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")


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
        assert text[c.start_char : c.end_char] == c.text
