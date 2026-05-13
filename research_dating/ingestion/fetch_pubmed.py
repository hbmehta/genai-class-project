"""Fetch PubMed abstracts for a faculty member via NCBI E-utilities."""

import json
import os
import time
from pathlib import Path
from xml.etree import ElementTree

import requests


_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_DELAY = 0.34  # ~3 req/s without API key; 0.1 with key


def _get_delay() -> float:
    return 0.1 if os.getenv("NCBI_API_KEY") else _DELAY


def _build_params(extra: dict) -> dict:
    params = {"retmode": "json", **extra}
    key = os.getenv("NCBI_API_KEY")
    if key:
        params["api_key"] = key
    return params


def fetch_abstracts(
    pubmed_query: str,
    faculty_id: str,
    max_results: int = 10,
    cache_dir: Path | None = None,
) -> list[dict]:
    """
    Search PubMed and return a list of abstract dicts.

    Each dict: {pmid, title, abstract, year, journal}
    Results are cached to cache_dir/{faculty_id}.json if provided.
    """
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{faculty_id}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    # Step 1: esearch — get PMIDs
    search_resp = requests.get(
        f"{_BASE}/esearch.fcgi",
        params=_build_params({
            "db": "pubmed",
            "term": pubmed_query,
            "retmax": max_results,
            "sort": "pub+date",
        }),
        timeout=15,
    )
    search_resp.raise_for_status()
    pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    time.sleep(_get_delay())

    if not pmids:
        return []

    # Step 2: efetch — get abstracts in XML
    fetch_resp = requests.get(
        f"{_BASE}/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            **({} if not os.getenv("NCBI_API_KEY") else {"api_key": os.getenv("NCBI_API_KEY")}),
        },
        timeout=30,
    )
    fetch_resp.raise_for_status()
    time.sleep(_get_delay())

    results = _parse_pubmed_xml(fetch_resp.text)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(results, indent=2))

    return results


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ElementTree.fromstring(xml_text)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_els = article.findall(".//AbstractText")
        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//PubDate/Year")
        if year_el is None:
            year_el = article.find(".//PubDate/MedlineDate")

        abstract_parts = [
            (el.get("Label", "") + ": " if el.get("Label") else "") + (el.text or "")
            for el in abstract_els
        ]
        abstract = " ".join(p for p in abstract_parts if p.strip())

        if not abstract:
            continue

        articles.append({
            "pmid": pmid_el.text if pmid_el is not None else "",
            "title": title_el.text if title_el is not None else "",
            "abstract": abstract,
            "year": year_el.text if year_el is not None else "",
            "journal": journal_el.text if journal_el is not None else "",
        })

    return articles
