"""
Dispatch chunking strategy per source type and build all_chunks.json.

Usage:
    cd research_dating
    uv run python -m chunking.chunk_pipeline
"""

import json
from pathlib import Path

from tqdm import tqdm

from chunking.header_chunker import chunk_by_headers
from chunking.sliding_window_chunker import chunk_sliding_window

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"


def _make_meta(profile: dict) -> dict:
    return {
        "id": profile["id"],
        "name": profile["name"],
        "department": profile["department"],
        "lab_url": profile.get("lab_url", ""),
    }


def chunk_faculty_profile(profile: dict) -> list[dict]:
    """Return all chunks for one faculty member across all sources."""
    meta = _make_meta(profile)
    chunks: list[dict] = []
    sources = profile.get("sources", {})

    # CV: header-based if sections dict, else sliding window
    cv = sources.get("cv")
    if isinstance(cv, dict) and cv:
        chunks.extend(chunk_by_headers(cv, meta))
    elif isinstance(cv, str) and cv.strip():
        chunks.extend(chunk_sliding_window(cv, meta, source_type="cv"))

    # Lab page: always sliding window
    lab_page = sources.get("lab_page", "")
    if lab_page:
        chunks.extend(chunk_sliding_window(lab_page, meta, source_type="lab_page"))

    # PubMed abstracts: each abstract is one chunk
    for i, paper in enumerate(sources.get("pubmed", [])):
        text = f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}"
        if text.strip():
            chunks.append({
                "text": text.strip(),
                "faculty_id": meta["id"],
                "faculty_name": meta["name"],
                "department": meta["department"],
                "source_type": "pubmed",
                "section": "",
                "lab_url": meta["lab_url"],
                "chunk_index": i,
                "pmid": paper.get("pmid", ""),
                "year": paper.get("year", ""),
            })

    # NIH grants: abstract as one chunk, project_terms as supplemental chunk
    for i, grant in enumerate(sources.get("nih_grants", [])):
        abstract_text = grant.get("abstract_text", "")
        title = grant.get("project_title", "")
        if abstract_text or title:
            text = f"{title}\n\n{abstract_text}".strip()
            chunks.append({
                "text": text,
                "faculty_id": meta["id"],
                "faculty_name": meta["name"],
                "department": meta["department"],
                "source_type": "nih_grant",
                "section": "",
                "lab_url": meta["lab_url"],
                "chunk_index": i,
                "fiscal_year": grant.get("fiscal_year", ""),
                "activity_code": grant.get("activity_code", ""),
            })

        # Project terms as a separate supplemental chunk (high-signal keywords)
        terms = grant.get("project_terms", "")
        if terms:
            chunks.append({
                "text": f"Research keywords for {meta['name']}: {terms}",
                "faculty_id": meta["id"],
                "faculty_name": meta["name"],
                "department": meta["department"],
                "source_type": "nih_grant_terms",
                "section": "",
                "lab_url": meta["lab_url"],
                "chunk_index": i,
            })

    return chunks


def main():
    profiles_path = PROCESSED_DIR / "faculty_profiles.json"
    if not profiles_path.exists():
        print(f"ERROR: {profiles_path} not found. Run ingestion first.")
        return

    profiles = json.loads(profiles_path.read_text())
    print(f"Chunking {len(profiles)} faculty profiles...")

    all_chunks: list[dict] = []
    source_counts: dict[str, int] = {}

    for profile in tqdm(profiles, desc="Chunking"):
        chunks = chunk_faculty_profile(profile)
        all_chunks.extend(chunks)
        for chunk in chunks:
            st = chunk["source_type"]
            source_counts[st] = source_counts.get(st, 0) + 1

    # Assign globally unique chunk IDs
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"{chunk['faculty_id']}_{chunk['source_type']}_{chunk['chunk_index']}_{i}"

    out_path = PROCESSED_DIR / "all_chunks.json"
    out_path.write_text(json.dumps(all_chunks, indent=2))

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("By source type:")
    for st, count in sorted(source_counts.items()):
        print(f"  {st}: {count}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
