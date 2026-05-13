"""ChromaDB interface for storing and querying faculty chunk embeddings."""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings

_COLLECTION_NAME = "faculty_chunks"
_METADATA_KEYS = [
    "faculty_id", "faculty_name", "department",
    "source_type", "section", "lab_url",
]


def get_chroma_client(persist_dir: str | None = None) -> chromadb.ClientAPI:
    if persist_dir is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    collection: chromadb.Collection,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Upsert chunks and their embeddings into ChromaDB."""
    if not chunks:
        return

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {k: c.get(k, "") for k in _METADATA_KEYS}
        for c in chunks
    ]

    # ChromaDB upsert in batches of 5000
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 15,
    where_filter: dict | None = None,
) -> dict:
    """
    Query the collection for the most similar chunks.

    Returns ChromaDB result dict with keys: ids, documents, metadatas, distances.
    Distances are cosine distances (0=identical, 2=opposite) since we use hnsw:space=cosine.
    Convert to similarity: sim = 1 - distance
    """
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    return collection.query(**kwargs)


def collection_stats(collection: chromadb.Collection) -> dict:
    count = collection.count()
    return {"total_chunks": count}
