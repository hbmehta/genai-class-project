"""
Embed all chunks and build the ChromaDB index.

Usage:
    cd research_dating
    uv run python -m vectordb.build_index
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from vectordb.chroma_store import get_chroma_client, get_or_create_collection, upsert_chunks, collection_stats
from vectordb.embed import embed_texts

load_dotenv()

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
_EMBED_BATCH = 128


def main():
    chunks_path = PROCESSED_DIR / "all_chunks.json"
    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run chunking pipeline first.")
        return

    chunks = json.loads(chunks_path.read_text())
    print(f"Loaded {len(chunks)} chunks.")

    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} texts (batch_size={_EMBED_BATCH})...")
    embeddings: list[list[float]] = []
    for i in tqdm(range(0, len(texts), _EMBED_BATCH), desc="Embedding batches"):
        batch = texts[i : i + _EMBED_BATCH]
        batch_embeddings = embed_texts(batch, input_type="document")
        embeddings.extend(batch_embeddings)

    print(f"Embedding complete. Upserting to ChromaDB...")
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    upsert_chunks(collection, chunks, embeddings)

    stats = collection_stats(collection)
    print(f"\nIndex built successfully.")
    print(f"  Total chunks in DB: {stats['total_chunks']}")

    # Faculty summary
    faculty_ids = list({c["faculty_id"] for c in chunks})
    print(f"  Faculty indexed: {len(faculty_ids)}")


if __name__ == "__main__":
    main()
