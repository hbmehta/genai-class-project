"""Model-as-judge prompt templates for automated evaluation."""

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a research mentor matching system at Johns Hopkins Bloomberg School of Public Health. A graduate student described their research interests, and the system returned a faculty match with an explanation and a draft outreach email.

Score this match on three dimensions using the rubric below.

## Relevance (1–3)
1 = No meaningful connection between faculty research and student interests
2 = Tangential connection; faculty works in the general area but specific focus doesn't align
3 = Strong, direct match on topic, method, or population

## Explanation Quality (1–3)
1 = Generic; could apply to any faculty; no specific citation of faculty work
2 = References faculty work but is vague or partially inaccurate
3 = Specific, grounded in faculty excerpts, clearly connects faculty work to student interests; no invented details

## Email Quality (1–3)
1 = Generic template; no personalization; no specific reference to faculty work
2 = Partially personalized; references general area but no specific project or paper
3 = Specific and professional; cites a concrete piece of faculty work; makes a clear ask

Return ONLY valid JSON — no prose, no markdown:
{"relevance": 1|2|3, "explanation_quality": 1|2|3, "email_quality": 1|2|3, "reasoning": "1-2 sentence justification"}"""


def build_judge_user_message(
    query: str,
    match: dict,
    faculty_excerpts: str,
) -> str:
    """Build the user message for the model-as-judge call."""
    return (
        f"Student research interests:\n{query}\n\n"
        f"Faculty matched: {match.get('faculty_name')} ({match.get('department')})\n\n"
        f"Explanation provided to student:\n{match.get('explanation', '')}\n\n"
        f"Draft outreach email:\n{match.get('email_draft', '')}\n\n"
        f"Relevant faculty profile excerpts (ground truth context):\n{faculty_excerpts}"
    )
