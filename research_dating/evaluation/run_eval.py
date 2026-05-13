"""
Run full evaluation: RAG pipeline + TF-IDF baseline on all test queries.

Usage:
    cd research_dating
    uv run python -m evaluation.run_eval

Outputs:
    evaluation/eval_results.json   — full results per query per match
    evaluation/eval_summary.json   — aggregate metrics
"""

import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from baseline.tfidf_baseline import TFIDFBaseline
from evaluation.cohen_kappa import print_agreement_report
from evaluation.judge_prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_message
from pipeline.rag_pipeline import run_pipeline

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

EVAL_DIR = Path("evaluation")
PROCESSED_DIR = Path("data/processed")

_MODEL = "claude-sonnet-4-6"


def run_model_judge(
    query: str,
    match: dict,
    faculty_chunks: list[dict],
    client: anthropic.Anthropic,
) -> dict:
    excerpts = "\n\n".join(c["text"] for c in faculty_chunks[:3])
    user_msg = build_judge_user_message(query, match, excerpts)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=256,
        temperature=0.0,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"relevance": 0, "explanation_quality": 0, "email_quality": 0, "reasoning": "parse error"}


def precision_at_k(
    returned_faculty_ids: list[str],
    ground_truth_ids: list[str],
    k: int,
) -> float:
    if not ground_truth_ids:
        return 0.0
    top_k = returned_faculty_ids[:k]
    hits = sum(fid in ground_truth_ids for fid in top_k)
    return hits / k


def main():
    # Load test queries
    queries_path = EVAL_DIR / "test_queries.json"
    test_queries = json.loads(queries_path.read_text())
    print(f"Loaded {len(test_queries)} test queries.")

    # Load chunks for TF-IDF baseline and faculty chunk lookup
    chunks_path = PROCESSED_DIR / "all_chunks.json"
    if not chunks_path.exists():
        print("ERROR: all_chunks.json not found. Run ingestion + chunking first.")
        return

    chunks = json.loads(chunks_path.read_text())
    chunks_by_faculty: dict[str, list] = {}
    for chunk in chunks:
        fid = chunk["faculty_id"]
        chunks_by_faculty.setdefault(fid, []).append(chunk)

    # Fit TF-IDF baseline
    print("Fitting TF-IDF baseline...")
    tfidf = TFIDFBaseline(chunks)
    tfidf.fit()

    # Anthropic client for model-as-judge
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []
    all_judge_scores = {"relevance": [], "explanation_quality": [], "email_quality": []}

    for tq in tqdm(test_queries, desc="Evaluating queries"):
        qid = tq["query_id"]
        query = tq["query_text"]
        ground_truth = tq.get("ground_truth_faculty", [])
        expected = tq.get("expected_behavior", "success")

        query_result = {
            "query_id": qid,
            "query_text": query,
            "specificity": tq.get("specificity"),
            "ground_truth_faculty": ground_truth,
            "rag": {},
            "tfidf": {},
        }

        # --- RAG pipeline ---
        rag_result = run_pipeline(query)
        query_result["rag"]["status"] = rag_result["status"]
        query_result["rag"]["max_similarity"] = rag_result.get("max_similarity", 0.0)
        query_result["rag"]["low_confidence"] = rag_result.get("low_confidence_warning", False)

        rag_matches = rag_result.get("matches") or []
        rag_faculty_ids = [m["faculty_id"] for m in rag_matches]

        query_result["rag"]["returned_faculty"] = rag_faculty_ids
        query_result["rag"]["precision_at_3"] = precision_at_k(rag_faculty_ids, ground_truth, 3)
        query_result["rag"]["precision_at_5"] = precision_at_k(rag_faculty_ids, ground_truth, 5)

        # Clarification correctness for low-specificity queries
        if expected == "clarification_needed":
            query_result["rag"]["clarification_correct"] = rag_result["status"] == "clarification_needed"

        # Model-as-judge scores for RAG matches
        judge_scores_for_query = []
        for match in rag_matches:
            fid = match.get("faculty_id", "")
            fchunks = chunks_by_faculty.get(fid, [])
            scores = run_model_judge(query, match, fchunks, client)
            judge_scores_for_query.append({
                "faculty_id": fid,
                "faculty_name": match.get("faculty_name"),
                **scores,
            })
            for dim in ["relevance", "explanation_quality", "email_quality"]:
                if scores.get(dim, 0) > 0:
                    all_judge_scores[dim].append(scores[dim])

        query_result["rag"]["judge_scores"] = judge_scores_for_query

        # --- TF-IDF baseline ---
        tfidf_result = tfidf.query(query, top_k=5)
        tfidf_ids = [r["faculty_id"] for r in tfidf_result]
        query_result["tfidf"]["returned_faculty"] = tfidf_ids
        query_result["tfidf"]["scores"] = [r["score"] for r in tfidf_result]
        query_result["tfidf"]["precision_at_3"] = precision_at_k(tfidf_ids, ground_truth, 3)
        query_result["tfidf"]["precision_at_5"] = precision_at_k(tfidf_ids, ground_truth, 5)

        results.append(query_result)

    # Save full results
    results_path = EVAL_DIR / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Aggregate summary
    rag_p3 = [r["rag"]["precision_at_3"] for r in results if r["rag"].get("status") != "clarification_needed"]
    rag_p5 = [r["rag"]["precision_at_5"] for r in results if r["rag"].get("status") != "clarification_needed"]
    tfidf_p3 = [r["tfidf"]["precision_at_3"] for r in results]
    tfidf_p5 = [r["tfidf"]["precision_at_5"] for r in results]

    clarification_checks = [
        r["rag"].get("clarification_correct")
        for r in results
        if "clarification_correct" in r["rag"]
    ]
    clarification_rate = sum(clarification_checks) / len(clarification_checks) if clarification_checks else None

    summary = {
        "rag": {
            "mean_precision_at_3": sum(rag_p3) / len(rag_p3) if rag_p3 else None,
            "mean_precision_at_5": sum(rag_p5) / len(rag_p5) if rag_p5 else None,
            "mean_relevance_judge": sum(all_judge_scores["relevance"]) / len(all_judge_scores["relevance"]) if all_judge_scores["relevance"] else None,
            "mean_explanation_judge": sum(all_judge_scores["explanation_quality"]) / len(all_judge_scores["explanation_quality"]) if all_judge_scores["explanation_quality"] else None,
            "mean_email_judge": sum(all_judge_scores["email_quality"]) / len(all_judge_scores["email_quality"]) if all_judge_scores["email_quality"] else None,
            "clarification_rate": clarification_rate,
        },
        "tfidf": {
            "mean_precision_at_3": sum(tfidf_p3) / len(tfidf_p3) if tfidf_p3 else None,
            "mean_precision_at_5": sum(tfidf_p5) / len(tfidf_p5) if tfidf_p5 else None,
        },
    }

    summary_path = EVAL_DIR / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nRAG Pipeline:")
    print(f"  Precision@3      : {summary['rag']['mean_precision_at_3']}")
    print(f"  Precision@5      : {summary['rag']['mean_precision_at_5']}")
    print(f"  Mean Relevance   : {summary['rag']['mean_relevance_judge']}")
    print(f"  Mean Explanation : {summary['rag']['mean_explanation_judge']}")
    print(f"  Mean Email       : {summary['rag']['mean_email_judge']}")
    print(f"  Clarification %  : {summary['rag']['clarification_rate']}")
    print(f"\nTF-IDF Baseline:")
    print(f"  Precision@3      : {summary['tfidf']['mean_precision_at_3']}")
    print(f"  Precision@5      : {summary['tfidf']['mean_precision_at_5']}")
    print(f"\nFull results saved to {results_path}")
    print(f"Summary saved to {summary_path}")
    print(
        "\nNEXT STEP: Collect human scores for a 5-query subset (q01–q05), then run "
        "cohen_kappa.print_agreement_report() to compute inter-rater agreement."
    )


if __name__ == "__main__":
    main()
