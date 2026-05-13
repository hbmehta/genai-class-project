"""Step 1: Check if a student query is specific enough before running RAG."""

import json
import os

import anthropic

_SYSTEM_PROMPT = """You are an assistant helping graduate students at Johns Hopkins Bloomberg School of Public Health find faculty research mentors. Your job is to assess whether a student's research interest query is specific enough to return useful faculty matches.

Assess specificity on three dimensions:
1. Topic specificity: Is there a specific disease, health outcome, or domain mentioned?
2. Methods specificity: Are specific methods, approaches, or data types mentioned?
3. Context specificity: Is there a population, geography, setting, or policy angle?

A query needs at least ONE dimension to be reasonably specific. If the query is purely generic (e.g., "I like public health", "I want to do research"), it is not specific enough.

Return ONLY valid JSON matching this schema — no prose, no markdown:
{"is_specific_enough": true/false, "follow_up_question": "question string or null", "reasoning": "brief internal note"}

If is_specific_enough is true, follow_up_question must be null.
If is_specific_enough is false, write one concise, friendly follow-up question to elicit more detail."""

_MODEL = "claude-sonnet-4-6"


def clarify_query(query: str, client: anthropic.Anthropic) -> dict:
    """
    Assess whether the query is specific enough for reliable RAG retrieval.

    Returns:
        {
            "is_specific_enough": bool,
            "follow_up_question": str | None,
            "reasoning": str
        }
    """
    response = client.messages.create(
        model=_MODEL,
        max_tokens=256,
        temperature=0.1,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Student query: {query}"}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: assume specific enough to avoid blocking the pipeline
        result = {
            "is_specific_enough": True,
            "follow_up_question": None,
            "reasoning": "JSON parse failed; defaulting to specific enough",
        }

    return result
