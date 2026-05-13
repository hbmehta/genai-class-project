"""Step 2: Embed the query and retrieve top-k relevant chunks from ChromaDB."""

import os

import chromadb

from vectordb.embed import embed_query

_MAX_CHUNKS_PER_FACULTY = 3


def retrieve_top_chunks(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 15,
    department_filter: str | None = None,
) -> dict:
    """
    Embed the query, query ChromaDB, and return deduplicated top-k chunks.

    Returns:
        {
            "chunks": list[dict],         # chunk dicts with text + metadata
            "similarities": list[float],  # cosine similarity per chunk (0-1)
            "low_confidence": bool,
            "max_similarity": float,
        }
    """
    low_confidence_threshold = float(
        os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.35")
    )

    query_emb = embed_query(query)

    where_filter = None
    if department_filter:
        where_filter = {"department": {"$eq": department_filter}}

    raw = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )

    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    # ChromaDB with hnsw:space=cosine returns cosine distances (0=identical)
    similarities = [1.0 - d for d in distances]

    # Build chunk list and deduplicate: cap at _MAX_CHUNKS_PER_FACULTY per faculty
    per_faculty_count: dict[str, int] = {}
    chunks = []
    sims_filtered = []

    for doc, meta, sim in zip(documents, metadatas, similarities):
        fid = meta.get("faculty_id", "")
        if per_faculty_count.get(fid, 0) >= _MAX_CHUNKS_PER_FACULTY:
            continue
        per_faculty_count[fid] = per_faculty_count.get(fid, 0) + 1
        chunks.append({"text": doc, **meta})
        sims_filtered.append(sim)

    max_sim = max(sims_filtered) if sims_filtered else 0.0

    return {
        "chunks": chunks,
        "similarities": sims_filtered,
        "low_confidence": max_sim < low_confidence_threshold,
        "max_similarity": max_sim,
    }
