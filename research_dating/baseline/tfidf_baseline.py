"""TF-IDF keyword baseline for comparison with the RAG pipeline."""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFBaseline:
    """
    Keyword-based faculty matching using TF-IDF cosine similarity.

    Used as the comparison baseline against the RAG + LLM pipeline.
    Evaluated on rank overlap with ground truth — no explanations generated.
    """

    def __init__(self, chunks: list[dict]):
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            stop_words="english",
        )
        self._matrix = None
        self._faculty_ids = [c["faculty_id"] for c in chunks]
        self._faculty_names = {
            c["faculty_id"]: c["faculty_name"] for c in chunks
        }

    def fit(self) -> None:
        """Build TF-IDF matrix over all chunk texts."""
        texts = [c["text"] for c in self._chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Return top-k faculty ranked by TF-IDF cosine similarity.

        Aggregates per-chunk scores to per-faculty scores by max pooling.

        Returns:
            [{"faculty_id": str, "faculty_name": str, "score": float}]
        """
        if self._matrix is None:
            raise RuntimeError("Call fit() before query()")

        query_vec = self._vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self._matrix)[0]

        # Max-pool: best chunk score per faculty
        faculty_scores: dict[str, float] = {}
        for fid, score in zip(self._faculty_ids, scores):
            faculty_scores[fid] = max(faculty_scores.get(fid, 0.0), float(score))

        ranked = sorted(faculty_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "faculty_id": fid,
                "faculty_name": self._faculty_names.get(fid, fid),
                "score": score,
            }
            for fid, score in ranked[:top_k]
        ]

    @classmethod
    def from_chunks_file(cls, chunks_path: str | Path) -> "TFIDFBaseline":
        """Load chunks from JSON file and return a fitted baseline."""
        chunks = json.loads(Path(chunks_path).read_text())
        baseline = cls(chunks)
        baseline.fit()
        return baseline
