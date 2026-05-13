"""
Embedding via Voyage AI (primary) with sentence-transformers fallback.

Voyage AI: https://docs.voyageai.com/docs/embeddings
Free tier: 200M tokens. Sign up at voyageai.com, set VOYAGE_API_KEY in .env.
"""

import os
import time
from typing import Literal

_VOYAGE_CLIENT = None
_ST_MODEL = None


def _get_voyage_client():
    global _VOYAGE_CLIENT
    if _VOYAGE_CLIENT is None:
        import voyageai
        _VOYAGE_CLIENT = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    return _VOYAGE_CLIENT


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _ST_MODEL


def _use_voyage() -> bool:
    if os.getenv("EMBEDDING_BACKEND", "").lower() == "sentence-transformers":
        return False
    return bool(os.getenv("VOYAGE_API_KEY"))


def embed_texts(
    texts: list[str],
    model: str | None = None,
    input_type: Literal["document", "query"] = "document",
    batch_size: int = 128,
) -> list[list[float]]:
    """
    Embed a list of texts. Uses Voyage AI if VOYAGE_API_KEY is set, else sentence-transformers.

    Args:
        texts: List of text strings to embed.
        model: Voyage model name (default from env EMBEDDING_MODEL or "voyage-3").
        input_type: "document" for corpus chunks, "query" for search queries.
        batch_size: Max texts per API call.
    """
    if not texts:
        return []

    if _use_voyage():
        return _embed_voyage(texts, model, input_type, batch_size)
    else:
        return _embed_sentence_transformers(texts)


def embed_query(query: str, model: str | None = None) -> list[float]:
    """Embed a single query string."""
    results = embed_texts([query], model=model, input_type="query")
    return results[0]


def _embed_voyage(
    texts: list[str],
    model: str | None,
    input_type: str,
    batch_size: int,
) -> list[list[float]]:
    client = _get_voyage_client()
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "voyage-3")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.embed(batch, model=model, input_type=input_type)
        all_embeddings.extend(result.embeddings)
        if i + batch_size < len(texts):
            time.sleep(0.05)

    return all_embeddings


def _embed_sentence_transformers(texts: list[str]) -> list[list[float]]:
    model = _get_st_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()
