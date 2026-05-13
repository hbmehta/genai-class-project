"""Step 3: Rank retrieved faculty using Claude Sonnet with structured JSON output."""

import json
from collections import defaultdict

import anthropic

_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are a research mentor advisor at Johns Hopkins Bloomberg School of Public Health. A new graduate student has described their research interests, and you have retrieved excerpts from faculty profiles.

Your task:
1. Identify the 3–5 faculty members whose research most closely aligns with the student's stated interests.
2. Rank them from strongest match (#1) to least strong match.
3. For each match, write a 3–4 sentence explanation. You MUST ground every sentence in the provided faculty excerpts. Do NOT invent research areas, papers, grants, or details not present in the excerpts.
4. For each match, identify 3–5 key terms that connect the student's interests to the faculty member's work.

Return ONLY valid JSON — no prose, no markdown fences — matching this schema exactly:
{
  "matches": [
    {
      "rank": 1,
      "faculty_name": "Full Name",
      "faculty_id": "faculty_id_string",
      "department": "Department name",
      "lab_url": "url or empty string",
      "explanation": "3-4 sentence explanation grounded in excerpts.",
      "key_overlap_terms": ["term1", "term2", "term3"]
    }
  ]
}"""


def _build_context(chunks: list[dict]) -> str:
    """Group chunks by faculty and format as labeled context blocks."""
    by_faculty: defaultdict[str, list] = defaultdict(list)
    for chunk in chunks:
        by_faculty[chunk["faculty_id"]].append(chunk)

    parts = []
    for fid, fchunks in by_faculty.items():
        name = fchunks[0].get("faculty_name", fid)
        dept = fchunks[0].get("department", "")
        lab = fchunks[0].get("lab_url", "")
        header = f"=== {name} | {dept} | faculty_id: {fid} | lab: {lab} ==="
        body = "\n\n".join(c["text"] for c in fchunks)
        parts.append(f"{header}\n{body}")

    return "\n\n---\n\n".join(parts)


def rank_faculty(
    query: str,
    chunks: list[dict],
    client: anthropic.Anthropic,
) -> dict:
    """
    Call Claude Sonnet to rank and explain faculty matches.

    Returns parsed dict with "matches" list, or empty matches on failure.
    """
    context = _build_context(chunks)
    user_message = (
        f"Student research interests:\n{query}\n\n"
        f"Faculty profile excerpts:\n\n{context}"
    )

    response = client.messages.create(
        model=_MODEL,
        max_tokens=2048,
        temperature=0.2,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
        if "matches" not in result:
            result = {"matches": []}
    except json.JSONDecodeError:
        result = {"matches": []}

    return result
