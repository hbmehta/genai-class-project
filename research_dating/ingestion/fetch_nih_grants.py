"""Fetch NIH Research Portfolio grants for a faculty member via NIH Reporter API."""

import json
from pathlib import Path

import requests


_NIH_REPORTER_URL = "https://api.reporter.nih.gov/v2/projects/search"


def fetch_grants(
    pi_name: str,
    faculty_id: str,
    max_results: int = 10,
    cache_dir: Path | None = None,
) -> list[dict]:
    """
    Search NIH Reporter for grants where pi_name is the PI.

    Returns list of dicts: {project_title, abstract_text, project_terms, fiscal_year, activity_code}
    Results are cached to cache_dir/{faculty_id}.json if provided.
    """
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{faculty_id}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    payload = {
        "criteria": {
            "pi_names": [{"any_name": pi_name}],
            "org_names": ["JOHNS HOPKINS UNIVERSITY"],
        },
        "offset": 0,
        "limit": max_results,
        "sort_field": "fiscal_year",
        "sort_order": "desc",
    }

    try:
        resp = requests.post(_NIH_REPORTER_URL, json=payload, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  NIH Reporter failed for {pi_name}: {e}")
        return []

    results_raw = resp.json().get("results", [])
    grants = []
    for r in results_raw:
        abstract = r.get("abstract_text") or ""
        title = r.get("project_title") or ""
        if not abstract and not title:
            continue
        grants.append({
            "project_title": title,
            "abstract_text": abstract,
            "project_terms": r.get("project_terms") or "",
            "fiscal_year": r.get("fiscal_year") or "",
            "activity_code": r.get("activity_code") or "",
        })

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(grants, indent=2))

    return grants
