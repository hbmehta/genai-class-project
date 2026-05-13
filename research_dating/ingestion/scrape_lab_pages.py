"""Scrape faculty lab/profile pages into clean plain text."""

import time
from pathlib import Path
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


_HEADERS = {
    "User-Agent": "ResearchDatingBot/1.0 (class project; contact hbmehta@jhu.edu)"
}

_NOISE_SELECTORS = [
    "nav", "header", "footer", "aside",
    ".sidebar", ".menu", ".navigation", ".breadcrumb",
    "#nav", "#header", "#footer", "#sidebar",
    "script", "style", "noscript",
]


def _can_fetch(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(_HEADERS["User-Agent"], url)
    except Exception:
        return True  # allow on error


def scrape_lab_page(url: str, output_path: Path | None = None) -> str:
    """
    Fetch and clean a faculty lab/profile page.

    Returns cleaned plain text. Optionally saves to output_path.
    Returns empty string if the page cannot be fetched.
    """
    if not _can_fetch(url):
        print(f"  robots.txt disallows: {url}")
        return ""

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            print(f"  Lab page blocked (403) — skipping: {url}")
        else:
            print(f"  Failed to fetch {url}: {e}")
        return ""
    except requests.RequestException as e:
        print(f"  Failed to fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "lxml")

    for selector in _NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    content_tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "li"])
    lines = []
    for tag in content_tags:
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text) > 20:
            lines.append(text)

    cleaned = "\n\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned, encoding="utf-8")

    time.sleep(2)
    return cleaned
