# ResearchDating Evaluation Rubric

## Relevance Score (1–3)

Score each returned faculty match against the student query:

| Score | Meaning |
|-------|---------|
| **1** | No meaningful connection between this faculty member's research and the student's stated interests. The match appears arbitrary or based on superficial keyword overlap. |
| **2** | Tangential connection. The faculty member works in the general area, but the specific focus (population, method, disease) does not align well. A student would be unlikely to find this faculty a productive fit. |
| **3** | Strong, direct match. The faculty member's research portfolio clearly overlaps with the student's interests on at least one specific dimension (topic, method, or population). A student would reasonably pursue this faculty as a mentor. |

## Explanation Quality Score (1–3)

Score the 3–4 sentence explanation provided for each match:

| Score | Meaning |
|-------|---------|
| **1** | Generic explanation that could apply to any faculty in the department. No specific citation of faculty work, grants, or publications. Could have been written without reading any faculty materials. |
| **2** | References some faculty work but is vague, partially inaccurate, or does not clearly connect the faculty's work to the student's specific interests. May include one specific claim but lacks depth. |
| **3** | Specific, grounded explanation that cites at least one concrete piece of faculty work (a project, publication area, or grant) and clearly articulates WHY this faculty is a good match for THIS student. Free of hallucinated or invented details. |

## Email Quality Score (1–3)

Score the draft outreach email generated for each match:

| Score | Meaning |
|-------|---------|
| **1** | Generic template with no personalization. Could have been addressed to any faculty member. No specific reference to faculty work. |
| **2** | Partially personalized. References the faculty's general area but does not cite a specific project, publication, or grant. Professional tone but lacks specificity. |
| **3** | Specific, warm, and professional. References a concrete piece of the faculty member's actual work (from the retrieved excerpts). Makes a clear ask. Would meaningfully distinguish this student from a generic cold email. |

---

## Scoring Process

1. For each test query, score all returned matches (up to 5) on Relevance.
2. Score the Explanation Quality for each match.
3. Score the Email Quality for each match.
4. Compare scores to model-as-judge scores on the same queries (5-query subset).
5. Compute Cohen's kappa for inter-rater agreement (human vs. model-as-judge).

## Aggregate Metrics

- **Precision@3**: Fraction of top-3 returned faculty with Relevance ≥ 3, averaged across queries.
- **Precision@5**: Same for top 5.
- **Mean Explanation Score**: Average explanation quality across all matches.
- **Mean Email Score**: Average email quality across all matches.
- **Clarification Rate**: Fraction of low-specificity queries (q11–q14) correctly triggering clarification.
