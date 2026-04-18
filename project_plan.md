# ResearchDating: An AI Agent for Faculty-Student Research Matching at Johns Hopkins Bloomberg School of Public Health

**Hemalkumar B. Mehta, MS, PhD**
Associate Professor of Epidemiology
Johns Hopkins Bloomberg School of Public Health

---

## 1. Project Title

**ResearchDating: An AI Agent for Faculty-Student Research Matching at Johns Hopkins Bloomberg School of Public Health**

---

## 2. Target User, Workflow, and Business Value

**Who the user is:**
Newly admitted master's-level graduate students (MHS, MSPH, MPH, ScM) at the Johns Hopkins Bloomberg School of Public Health (BSPH) in their first semester, who need to identify 3–5 faculty members aligned with their research interests to pursue as potential advisors or collaborators for their thesis or capstone projects. PhD students are explicitly out of scope — they typically arrive with a clearer research direction and a more defined path to finding an advisor before enrollment. For the pilot project, we will focus on students from two departments — Epidemiology and Biostatistics — with the intent to expand to all BSPH departments in future iterations.

> **Johns Hopkins Bloomberg School of Public Health** has 873 Primary Faculty, 374 Affiliated Faculty and 80+ Centers & Institutes. The school admits nearly 3000 students each year. It includes several departments such as Biochemistry and Molecular Biology, Biostatistics, Environmental Health and Engineering, Epidemiology, Health, Behavior and Society, Health Policy and Management, International Health, Mental Health, Molecular Microbiology and Immunology, and Population, Family and Reproductive Health.

**What recurring task or decision you are improving:**
The task is faculty discovery and research interest matching — a one-time but high-stakes decision that every incoming master's student must make, typically within their first weeks or months of enrollment. Although students typically find mentors within the department, they often look for expertise and collaborations outside the department.

**Where the workflow begins and ends:**
The workflow begins when a student enters a free-text description of their research interests (e.g., topics, methods, populations, or diseases they want to study). It ends when the agent returns a ranked list of 3–5 faculty members with a brief explanation of why each is a good match, drawn from their CVs, lab pages, publications, and directory profiles. Where relevant, the agent also surfaces highlights from a faculty member's recent work that connect directly to the student's stated interests. For each faculty match, the agent also drafts a personalized outreach email that the student can copy, edit, and send directly.

**Why better performance on this workflow matters, and to whom:**
Currently, students rely on manual browsing of the BSPH faculty website, word-of-mouth from seniors, and Google searches — a process that is time-consuming, incomplete, and heavily dependent on social networks that newly admitted students have not yet built. This disadvantages students who lack informal connections to the school. Some departments partially address this by asking faculty to submit descriptions of ongoing research projects via survey and circulating that to students, but this is ad hoc, inconsistent across departments, and still places the matching burden on the student. The current manual process is inefficient and may not match the right students with the right faculty, especially those in other departments. Given the really large size of the school, it is not unreasonable for students to fail to find the right faculty. A better matching process benefits students (faster, more equitable access to the right advisor), faculty (more relevant inbound interest from students), and the school (better advising outcomes, reduced mismatch attrition).

---

## 3. Problem Statement and GenAI Fit

**The exact task:**
Given a newly admitted student's free-text description of their research interests, the system retrieves and ranks 3–5 BSPH faculty members whose research portfolios are the best match, with a plain-language explanation for each recommendation.

**What part of the workflow benefits from language models:**
Faculty research profiles are rich, unstructured text — CVs, personal webpages, NIH grants, and PubMed abstracts written in varied formats and academic language. Matching a student's expressed interests to this corpus requires semantic understanding, not keyword search. A language model can interpret that a student interested in "diabetes prevention in low-income communities" maps to faculty working on "cardiometabolic health disparities" or "behavioral interventions in underserved populations" — connections a keyword search would miss entirely.

**Why a simpler non-GenAI tool would not be enough:**
A keyword-based search of the faculty directory would miss semantic overlap between how students describe their interests and how faculty describe their work. A static dropdown or filter system cannot handle the nuance and variety of research interest descriptions that incoming students bring from diverse academic backgrounds. Crucially, many first-semester students do not yet have precise research vocabulary — they may know broadly what they care about but lack the domain-specific terminology to run effective searches. An LLM-powered system allows students to describe their interests in plain conversational language and still receive semantically accurate matches, lowering the barrier to entry for students who are new to public health research.

---

## 4. Planned System Design and Baseline

**System architecture:**
ResearchDating uses a RAG pipeline over a curated corpus of faculty profiles from two pilot departments — Biostatistics and Epidemiology. This focused scope allows for rigorous evaluation and quality control, with the intent to expand to all BSPH departments and eventually the broader Johns Hopkins University in future iterations. Before retrieval, the system runs a lightweight query clarification step — a low-cost LLM call that classifies whether the student's input is specific enough to retrieve meaningful matches. If the query falls below a specificity threshold (e.g., "I want to do global health stuff"), the system returns a single targeted follow-up question ("Could you tell me more about the population, disease area, or methods you are interested in?") before running the full RAG pipeline. This ensures that vague inputs — one of the most common failure modes — are caught early rather than producing low-quality matches downstream. Each faculty member's data — drawn from their CV (PDF), lab webpage, NIH grants, and top PubMed publications — is chunked, embedded, and stored in a vector database. When a student submits a free-text description of their research interests, the system retrieves the most semantically relevant faculty chunks, then passes them to an LLM to generate a ranked list of 3–5 faculty matches with plain-language explanations.

**Course concepts integrated:**

1. **Retrieval-Augmented Generation (RAG) — Week 4:** Faculty CVs, lab pages, NIH grants, and PubMed abstracts will be chunked using a hybrid strategy — header-based chunking where section structure is present (e.g., Research Interests, Selected Publications, Current Projects), with a fallback to sliding window chunking (~300-500 tokens with overlap) for unstructured pages. PubMed abstracts will be chunked as individual units. NIH grant Specific Aims pages will be extracted and chunked separately as the most information-dense section. Chunks will be embedded using both `text-embedding-3-small` and `text-embedding-3-large`, with a comparison on the evaluation test set to select the better-performing model. For each student query, the top-k most relevant chunks will be retrieved and passed as context to the LLM. This ensures the model's recommendations are grounded in actual faculty content rather than hallucinated from training data.

2. **Evaluation design: rubrics, test sets, baselines, model-as-judge — Week 6:** A test set of 10–15 simulated student queries will be constructed, each with a ground-truth set of relevant faculty validated by the project author as a domain expert. Matches will be scored on relevance, explanation quality, and rank accuracy. A model-as-judge will score responses, supplemented by manual spot checks.

3. **Anatomy of an LLM call: system prompts, temperature, output constraints, structured outputs — Weeks 2–3:** The system prompt will explicitly instruct the LLM on its role (a research matching assistant for BSPH students), the ranking criteria (relevance of faculty research to student interests), and the required output format. Temperature will be set low (e.g., 0.2) to ensure consistent, reproducible rankings across similar queries. The LLM will be constrained to return structured JSON output — each match containing faculty name, rank, and a 3–4 sentence explanation — which is then rendered cleanly on the webpage. This design prevents the model from going off-format and makes the output easy to validate programmatically.

**Baseline:**
The simpler alternative is a keyword search over the same faculty corpus — a student's query terms are matched against faculty profile text using TF-IDF or simple string matching, returning the top results by overlap score. This represents the "just use search" alternative and sets a meaningful bar for whether RAG + LLM actually adds value.

**The webpage:**
The user sees a simple webpage with a single text box. They type a description of their research interests in plain conversational language (e.g., "I want to study mental health interventions for adolescents in urban settings"). The system uses two separate LLM calls — the first ranks faculty and generates match explanations, the second drafts a personalized outreach email for each match using the faculty profile, student query, and match explanation as context. The email prompt explicitly instructs the model to only reference research, papers, or grants present in the retrieved faculty context and not to invent or infer details. The webpage returns a ranked list of 3–5 faculty, each with a 3–4 sentence explanation and a draft outreach email the student can copy, edit, and send. The student can also click through to the faculty member's profile or lab page.

> **Disclaimer displayed to all users:** *ResearchDating helps you find faculty whose research aligns with your interests. It does not assess mentorship style, lab culture, or advisor availability. We strongly encourage you to speak with current students in any lab before reaching out.*

---

## 5. Evaluation Plan

**What success looks like:**
The system successfully identifies faculty whose research genuinely aligns with a student's stated interests, ranks them in a sensible order, and provides explanations specific enough that a student could use them to write a personalized outreach email. A successful match is one that a domain expert (faculty member or senior student) would independently agree is relevant.

**What you will measure:**
- **Relevance accuracy:** For each test query, what fraction of the returned 3–5 faculty are genuinely relevant matches, as judged against a ground-truth set
- **Rank quality:** Are the best matches appearing at the top of the list, or are they buried
- **Explanation quality:** Are the explanations specific, grounded in actual faculty work, and free of hallucinated details — scored on a 1–3 rubric (poor / adequate / strong)
- **Draft email quality:** Is the generated outreach email personalized, professional, and grounded in the faculty member's actual work — scored on a 1–3 rubric
- **Baseline comparison:** Does the RAG + LLM system outperform TF-IDF keyword search on relevance accuracy and explanation quality
- **Latency and cost:** How long does a query take, and what is the approximate API cost per query — relevant for assessing real-world deployability

**Test set:**
A set of 10–15 simulated student queries written to reflect realistic variation in how incoming master's students describe their interests — ranging from precise ("I want to study air pollution exposure and respiratory outcomes in children") to vague ("I'm interested in health and communities"). Queries will be scoped to research interests relevant to Biostatistics and Epidemiology faculty. Ground-truth relevant faculty for each query will be established by the project author as a domain expert, drawing on knowledge of the BSPH faculty. Where possible, 1–2 additional faculty validators will be consulted to reduce single-rater bias.

**Baseline comparison:**
Each test query will be run through both the ResearchDating RAG pipeline and the TF-IDF keyword baseline. Results will be compared on relevance accuracy and explanation quality. The model-as-judge will score both sets of outputs using the same rubric, and spot checks will be performed manually to validate the judge's scoring. Cosine similarity scores will be logged for all test queries to empirically calibrate the low-confidence retrieval threshold — rather than setting it arbitrarily, the threshold will be derived from the observed distribution of similarity scores for true matches versus non-matches in the test set.

---

## 6. Example Inputs and Failure Cases

**Example inputs:**

1. **Specific and well-formed:** "I want to study the impact of air pollution on cardiovascular disease outcomes in elderly populations in urban settings." — A precise query with clear topic, population, and setting that should produce reliable, high-confidence matches.

2. **Methods-focused:** "I am interested in using machine learning and electronic health records to study chronic disease." — Tests whether the system can match on methodological preferences rather than just topic area, since some faculty may describe their methods prominently while others do not.

3. **Policy and practice oriented:** "I care about health policy and want to work on improving access to mental health services for underserved communities." — Tests matching across the intersection of policy, health equity, and a specific disease area.

4. **Vague and conversational:** "I just want to do something related to global health and maybe infectious disease. I'm not sure exactly what yet." — Represents the realistic case of a student who is early in their thinking. Tests whether the system can still return useful matches or appropriately asks for clarification.

5. **Interdisciplinary and hard to place:** "I have a background in economics and want to apply causal inference methods to study the effects of social policy on health outcomes." — Tests whether the system can match across departmental boundaries and identify faculty whose work spans health economics, epidemiology, and policy.

**Anticipated failure cases:**

1. **Faculty with thin or outdated digital footprints:** Some faculty may have sparse CVs, rarely updated lab pages, and few recent PubMed publications — particularly senior faculty or those who are primarily educators rather than active researchers. The system may consistently underrank or miss them entirely, even when they would be a strong match, simply because their corpus is too thin to surface in retrieval.

2. **Highly vague queries with no retrievable signal:** When a student's input is too broad or generic (e.g., "I want to do public health research"), the retrieval step may return a diffuse, low-confidence set of chunks that leads the LLM to produce poorly justified or arbitrary rankings. The system needs a graceful way to handle this — either by asking the student a clarifying follow-up question or by explicitly flagging low confidence in the output.

3. **Hallucinated or over-interpreted explanations:** Even with RAG grounding, the LLM may generate explanations that overstate the connection between a student's interests and a faculty member's work, or incorrectly attribute a research area to a faculty member based on weak retrieval. This is a trust and reliability risk, particularly if students act on recommendations without verifying them independently.

---

## 7. Risks and Governance

**Where the system could fail:**
- **Retrieval gaps:** If a faculty member's research is not well represented in the curated corpus — due to sparse CVs, missing publications, or outdated lab pages — the system will systematically underrank them, creating an invisible bias in who gets recommended
- **Hallucinated explanations:** The LLM may generate plausible-sounding but factually incorrect explanations, attributing research areas or publications to faculty members that do not accurately reflect their work
- **Query misinterpretation:** For vague or ambiguous student inputs, the system may retrieve weakly relevant chunks and produce low-quality matches without signaling its own uncertainty to the user
- **Corpus staleness:** Faculty research interests evolve. A corpus built at one point in time will drift out of date, meaning the system could recommend faculty who have moved on from a research area or miss faculty who have recently entered one

**Where the system should not be trusted:**
- ResearchDating should not be treated as a definitive or authoritative advisor matching system. It is a discovery and exploration tool, not a placement system
- The system should not be used to make or communicate formal advising assignments on behalf of the school or any department
- Recommendations should not be presented to students as guaranteed matches — the final decision to reach out, meet, and form an advising relationship must remain with the student and faculty member
- The system should not be the sole resource students rely on. It should be explicitly positioned as a complement to, not a replacement for, department orientation sessions, senior student networks, and faculty office hours

**Controls, refusal rules, and human-review boundaries:**
- The webpage should display a clear disclaimer stating that recommendations are AI-generated, based on publicly available faculty information, and should be verified by the student before acting on them
- The system should refuse or flag queries that are off-topic, abusive, or unrelated to academic research matching (e.g., requests for personal information about faculty)
- Low-confidence retrievals — where the top chunks fall below a similarity threshold — should trigger a visible warning to the student that the match quality may be limited, and prompt them to refine their query
- A faculty opt-out mechanism should be considered for any real deployment, allowing faculty to request exclusion from the matching corpus

**Data, privacy, and cost concerns:**
- All faculty data used must be publicly available (CVs, lab pages, PubMed, NIH Reporter). No private or proprietary institutional data should be ingested without explicit permission, in line with course ground rules
- No student data should be stored or logged beyond the current session. Students should not need to create accounts or submit personally identifying information to use the system
- API costs should remain modest given the project scope — a curated corpus of 20–50 faculty and a test set of 10–15 queries will involve minimal embedding and inference calls. Costs should be estimated and monitored during development

---

## 8. Plan for the Week 6 Check-in

**What part of the app I expect to have running:**
By Week 6, the core RAG pipeline will be functional end-to-end for the Biostatistics and Epidemiology pilot departments. This includes a curated corpus of faculty profiles ingested, chunked, and embedded into a vector database. The webpage will accept a student's free-text query and return a ranked list of 3–5 faculty matches with plain-language explanations and a draft personalized outreach email for each match. The structured JSON output and system prompt design will be in place, with temperature and output constraints configured.

**What part of the evaluation I expect to have in place:**
A first-pass test set of 10–15 simulated student queries will be constructed, with ground-truth relevant faculty identified for each query. At least one scoring rubric will be operational — covering relevance accuracy and explanation quality — and the model-as-judge will have been run on at least a subset of the test queries. Manual spot checks on a sample of outputs will have been completed.

**Which comparison against the baseline I expect to be able to run:**
The TF-IDF keyword baseline will be implemented and run against the same 10–15 test queries. A side-by-side comparison of relevance accuracy between the RAG + LLM system and the keyword baseline will be available, giving an early signal on whether the GenAI approach is adding measurable value over the simpler alternative.

**What will not yet be complete:**
Full red-teaming and adversarial testing will likely still be in progress. The corpus may not yet include NIH grant data for all faculty. Final latency and cost measurements will be preliminary. These will be completed between Week 6 and the final submission.
