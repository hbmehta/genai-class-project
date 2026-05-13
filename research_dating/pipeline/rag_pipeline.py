"""
Main RAG pipeline orchestrator.

Ties together query clarification → retrieval → ranking → email drafting.
"""

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv

from pipeline.email_drafter import draft_outreach_email
from pipeline.query_clarifier import clarify_query
from pipeline.ranker import rank_faculty
from pipeline.retriever import retrieve_top_chunks
from vectordb.chroma_store import get_chroma_client, get_or_create_collection

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

_anthropic_client: anthropic.Anthropic | None = None
_chroma_collection: chromadb.Collection | None = None


def get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def get_collection() -> chromadb.Collection:
    global _chroma_collection
    if _chroma_collection is None:
        client = get_chroma_client()
        _chroma_collection = get_or_create_collection(client)
    return _chroma_collection


def run_pipeline(
    query: str,
    department_filter: str | None = None,
) -> dict:
    """
    Run the full ResearchDating pipeline for a student query.

    Args:
        query: Student's free-text research interest description.
        department_filter: Optional department name to restrict results.

    Returns:
        {
            "status": "clarification_needed" | "low_confidence" | "success",
            "follow_up_question": str | None,
            "matches": list[dict] | None,
            "low_confidence_warning": bool,
            "max_similarity": float,
        }

        Each match dict includes: rank, faculty_name, faculty_id, department,
        lab_url, explanation, key_overlap_terms, email_draft.
    """
    anthropic_client = get_anthropic_client()
    collection = get_collection()

    # Step 1: Query clarification
    clarification = clarify_query(query, anthropic_client)
    if not clarification.get("is_specific_enough", True):
        return {
            "status": "clarification_needed",
            "follow_up_question": clarification.get("follow_up_question"),
            "matches": None,
            "low_confidence_warning": False,
            "max_similarity": 0.0,
        }

    # Step 2: Retrieval
    retrieval = retrieve_top_chunks(
        query,
        collection,
        n_results=15,
        department_filter=department_filter,
    )
    chunks = retrieval["chunks"]
    max_sim = retrieval["max_similarity"]
    low_confidence = retrieval["low_confidence"]

    if not chunks:
        return {
            "status": "low_confidence",
            "follow_up_question": None,
            "matches": [],
            "low_confidence_warning": True,
            "max_similarity": 0.0,
        }

    # Step 3: Ranking
    ranking_result = rank_faculty(query, chunks, anthropic_client)
    matches = ranking_result.get("matches", [])

    # Step 4: Email drafting — parallel across matches
    chunks_by_faculty: defaultdict[str, list] = defaultdict(list)
    for chunk in chunks:
        chunks_by_faculty[chunk["faculty_id"]].append(chunk)

    def _draft(match: dict) -> tuple[int, str]:
        fid = match.get("faculty_id", "")
        faculty_chunks = chunks_by_faculty.get(fid, [])
        return match["rank"], draft_outreach_email(query, match, faculty_chunks, anthropic_client)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_draft, m): m for m in matches}
        rank_to_email = {}
        for future in as_completed(futures):
            rank, email = future.result()
            rank_to_email[rank] = email

    for match in matches:
        match["email_draft"] = rank_to_email[match["rank"]]

    status = "low_confidence" if low_confidence else "success"

    return {
        "status": status,
        "follow_up_question": None,
        "matches": matches,
        "low_confidence_warning": low_confidence,
        "max_similarity": max_sim,
    }
