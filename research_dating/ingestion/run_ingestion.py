"""
Orchestrate ingestion for all faculty in faculty_index.json.

Usage:
    cd research_dating
    uv run python -m ingestion.run_ingestion
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ingestion.fetch_nih_grants import fetch_grants
from ingestion.fetch_pubmed import fetch_abstracts
from ingestion.parse_cvs import extract_cv_text
from ingestion.scrape_lab_pages import scrape_lab_page

import os

load_dotenv()

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# CV folder can be overridden via FACULTY_CVS_DIR in .env
_CV_DIR = Path(os.getenv("FACULTY_CVS_DIR", str(RAW_DIR / "faculty_cvs")))


def build_faculty_profile(faculty: dict) -> dict:
    fid = faculty["id"]
    print(f"\nProcessing: {faculty['name']} ({fid})")

    # CV
    cv_result: dict | str = {}
    cv_file = faculty.get("cv_file")
    if cv_file:
        cv_path = _CV_DIR / cv_file
        if cv_path.exists():
            print(f"  Parsing CV: {cv_path.name}")
            cv_result = extract_cv_text(cv_path)
        else:
            print(f"  CV not found: {cv_path}")

    # Lab page
    lab_text = ""
    lab_url = faculty.get("lab_url", "")
    if lab_url:
        lab_cache = RAW_DIR / "lab_pages" / f"{fid}.txt"
        if lab_cache.exists():
            print(f"  Loading cached lab page")
            lab_text = lab_cache.read_text(encoding="utf-8")
        else:
            print(f"  Scraping lab page: {lab_url}")
            lab_text = scrape_lab_page(lab_url, output_path=lab_cache)

    # PubMed
    pubmed_results = []
    pubmed_query = faculty.get("pubmed_query", "")
    if pubmed_query:
        print(f"  Fetching PubMed abstracts")
        pubmed_results = fetch_abstracts(
            pubmed_query,
            fid,
            cache_dir=RAW_DIR / "pubmed",
        )
        print(f"    -> {len(pubmed_results)} abstracts")

    # NIH grants
    grants = []
    nih_pi_name = faculty.get("nih_pi_name", "")
    if nih_pi_name:
        print(f"  Fetching NIH grants")
        grants = fetch_grants(
            nih_pi_name,
            fid,
            cache_dir=RAW_DIR / "nih_grants",
        )
        print(f"    -> {len(grants)} grants")

    return {
        "id": fid,
        "name": faculty["name"],
        "department": faculty["department"],
        "email": faculty.get("email", ""),
        "lab_url": lab_url,
        "sources": {
            "cv": cv_result,
            "lab_page": lab_text,
            "pubmed": pubmed_results,
            "nih_grants": grants,
        },
    }


def main():
    index_path = DATA_DIR / "faculty_index.json"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found. Create it first.")
        return

    faculty_list = json.loads(index_path.read_text())
    print(f"Found {len(faculty_list)} faculty in index.")

    profiles = []
    for faculty in tqdm(faculty_list, desc="Ingesting faculty"):
        profile = build_faculty_profile(faculty)
        profiles.append(profile)

    out_path = PROCESSED_DIR / "faculty_profiles.json"
    out_path.write_text(json.dumps(profiles, indent=2))
    print(f"\nSaved {len(profiles)} profiles to {out_path}")

    # Summary
    missing_cv = [p["name"] for p in profiles if not p["sources"]["cv"]]
    missing_lab = [p["name"] for p in profiles if not p["sources"]["lab_page"]]
    print(f"\nSummary:")
    print(f"  Missing CV:       {len(missing_cv)}: {missing_cv}")
    print(f"  Missing lab page: {len(missing_lab)}: {missing_lab}")


if __name__ == "__main__":
    main()
