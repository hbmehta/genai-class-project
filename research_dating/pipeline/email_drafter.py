"""Step 4: Draft a personalized outreach email for each faculty match."""

import anthropic

_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are helping a new graduate student at Johns Hopkins Bloomberg School of Public Health write a professional outreach email to a potential research mentor.

Write a concise, warm, and professional email that:
- Opens with a specific, genuine reference to one piece of the faculty member's research (from the excerpts ONLY)
- Briefly introduces the student's research interests (1-2 sentences)
- Makes a clear, specific ask: a 20-minute meeting to discuss potential research opportunities
- Closes professionally
- Is 150–200 words total

IMPORTANT CONSTRAINTS:
- Only reference research, papers, grants, or projects explicitly present in the faculty excerpts
- Do NOT invent or infer any details not in the excerpts
- Use "Dear Professor [Last Name]" as the salutation
- Leave "[Your Name]" as a placeholder for the student's signature
- Do not include a subject line — only the email body"""


def draft_outreach_email(
    query: str,
    faculty_match: dict,
    faculty_chunks: list[dict],
    client: anthropic.Anthropic,
) -> str:
    """
    Draft a personalized outreach email for one faculty match.

    Args:
        query: Student's original research interest query.
        faculty_match: One entry from the ranker output (includes explanation).
        faculty_chunks: Chunks for this specific faculty member.
        client: Anthropic client.

    Returns:
        Plain-text email body string.
    """
    faculty_context = "\n\n".join(c["text"] for c in faculty_chunks)
    last_name = faculty_match.get("faculty_name", "Professor").split()[-1]

    user_message = (
        f"Student's research interests:\n{query}\n\n"
        f"Faculty member: {faculty_match.get('faculty_name')} "
        f"({faculty_match.get('department')})\n\n"
        f"Why this faculty is a good match:\n{faculty_match.get('explanation', '')}\n\n"
        f"Faculty profile excerpts:\n\n{faculty_context}\n\n"
        f"Write the email to Professor {last_name}."
    )

    response = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        temperature=0.5,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text.strip()
