"""Header-based chunker for structured CV sections."""

from chunking.sliding_window_chunker import chunk_sliding_window

_MAX_SECTION_TOKENS = 500


def chunk_by_headers(
    sections: dict[str, str],
    faculty_meta: dict,
) -> list[dict]:
    """
    Convert a CV section dict into a flat list of chunk dicts.

    Sections that exceed _MAX_SECTION_TOKENS are further split with the
    sliding window chunker to keep each chunk within embedding model limits.
    """
    chunks = []
    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue

        sub_chunks = chunk_sliding_window(
            section_text,
            faculty_meta,
            source_type="cv",
            chunk_size=_MAX_SECTION_TOKENS,
            overlap=80,
            section=section_name,
        )
        chunks.extend(sub_chunks)

    return chunks
