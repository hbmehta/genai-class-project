"""Token-based sliding window chunker for unstructured text."""

import nltk
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _build_chunk_dict(
    text: str,
    faculty_meta: dict,
    source_type: str,
    index: int,
    section: str = "",
) -> dict:
    return {
        "text": text,
        "faculty_id": faculty_meta["id"],
        "faculty_name": faculty_meta["name"],
        "department": faculty_meta["department"],
        "source_type": source_type,
        "section": section,
        "lab_url": faculty_meta.get("lab_url", ""),
        "chunk_index": index,
    }


def chunk_sliding_window(
    text: str,
    faculty_meta: dict,
    source_type: str,
    chunk_size: int = 400,
    overlap: int = 80,
    section: str = "",
) -> list[dict]:
    """
    Split text into overlapping token-window chunks, breaking on sentence boundaries.

    Returns list of chunk dicts with faculty metadata attached.
    """
    _ensure_nltk()

    if not text.strip():
        return []

    # If the whole text fits in one chunk, return it as-is
    if _count_tokens(text) <= chunk_size:
        return [_build_chunk_dict(text.strip(), faculty_meta, source_type, 0, section)]

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        # If a single sentence exceeds chunk_size, add it alone
        if sentence_tokens > chunk_size:
            if current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append(_build_chunk_dict(chunk_text, faculty_meta, source_type, chunk_index, section))
                chunk_index += 1
                # Keep overlap sentences for next window
                overlap_sentences = _build_overlap(current_sentences, overlap)
                current_sentences = overlap_sentences
                current_tokens = sum(_count_tokens(s) for s in current_sentences)

            chunks.append(_build_chunk_dict(sentence.strip(), faculty_meta, source_type, chunk_index, section))
            chunk_index += 1
            current_sentences = []
            current_tokens = 0
            continue

        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append(_build_chunk_dict(chunk_text, faculty_meta, source_type, chunk_index, section))
            chunk_index += 1
            # Keep overlap portion for next window
            overlap_sentences = _build_overlap(current_sentences, overlap)
            current_sentences = overlap_sentences
            current_tokens = sum(_count_tokens(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        chunks.append(_build_chunk_dict(chunk_text, faculty_meta, source_type, chunk_index, section))

    return chunks


def _build_overlap(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Return trailing sentences that together fit within overlap_tokens."""
    result = []
    total = 0
    for sentence in reversed(sentences):
        t = _count_tokens(sentence)
        if total + t > overlap_tokens:
            break
        result.insert(0, sentence)
        total += t
    return result
