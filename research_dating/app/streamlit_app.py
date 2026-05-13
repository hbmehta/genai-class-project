"""
ResearchDating — Streamlit web application.

Usage:
    cd research_dating
    uv run streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Allow imports from the project root when run via streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from app.components.faculty_card import render_faculty_card
from pipeline.rag_pipeline import run_pipeline
from vectordb.chroma_store import get_chroma_client, get_or_create_collection, collection_stats, upsert_chunks
from vectordb.embed import embed_texts

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)


@st.cache_resource(show_spinner=False)
def _ensure_index_built():
    """Build ChromaDB index from all_chunks.json if not already present."""
    import json
    chunks_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "all_chunks.json"
    if not chunks_path.exists():
        return False, "all_chunks.json not found."

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    stats = collection_stats(collection)

    if stats["total_chunks"] > 0:
        return True, f"Index ready ({stats['total_chunks']} chunks)."

    # Build index
    chunks = json.loads(chunks_path.read_text())
    texts = [c["text"] for c in chunks]
    batch_size = 128
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_embeddings = embed_texts(texts[i:i + batch_size], input_type="document")
        embeddings.extend(batch_embeddings)
    upsert_chunks(collection, chunks, embeddings)
    return True, f"Index built ({len(chunks)} chunks)."

_DISCLAIMER = (
    "Results are based on publicly available faculty CVs, publications, and grant abstracts. "
    "This tool does not assess mentorship style, lab culture, or advisor availability."
)

_PLACEHOLDER = (
    "e.g., I want to study the impact of air pollution on cardiovascular outcomes "
    "in elderly urban populations, using observational data and causal inference methods."
)

_DEPARTMENTS = ["All departments", "Epidemiology", "Biostatistics"]


def _init_session_state():
    defaults = {
        "clarification_pending": False,
        "follow_up_question": None,
        "original_query": "",
        "results": None,
        "last_query": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _render_results(matches: list[dict], low_confidence: bool, max_sim: float):
    if low_confidence:
        st.warning(
            f"Match confidence is low (similarity: {max_sim:.2f}). "
            "Consider refining your query with a more specific topic, population, or method."
        )

    if not matches:
        st.info("No faculty matches found. Try broadening your query or removing the department filter.")
        return

    st.success(f"Found {len(matches)} faculty match{'es' if len(matches) != 1 else ''}.")
    for i, match in enumerate(matches):
        render_faculty_card(match, i)


_CSS = """
<style>
/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #68ACE5;
}
[data-testid="stSidebar"] * {
    color: #002D72 !important;
}
[data-testid="stSidebar"] a {
    color: #002D72 !important;
    font-weight: 600;
}
[data-testid="stSidebar"] hr {
    border-color: #002D7233;
}

/* ── Primary button ──────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background-color: #0077D8;
    border: none;
    color: white;
}
.stButton > button[kind="primary"]:hover {
    background-color: #002D72;
    color: white;
}

/* ── Page title ──────────────────────────────────────────── */
h1 {
    color: #002D72;
}

/* ── Info / disclaimer box ───────────────────────────────── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-left: 4px solid #0077D8;
    background-color: #EBF4FF;
}

/* ── Success box ─────────────────────────────────────────── */
div[data-testid="stAlert"] .st-success {
    border-left: 4px solid #275E3D;
}

/* ── Faculty card headings ───────────────────────────────── */
[data-testid="stExpander"] summary {
    color: #002D72;
    font-weight: 600;
}

/* ── Divider ─────────────────────────────────────────────── */
hr {
    border-color: #68ACE533;
}
</style>
"""


def main():
    st.set_page_config(
        page_title="ResearchDating | BSPH",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CSS, unsafe_allow_html=True)
    _init_session_state()

    # Build index on first run if needed (Streamlit Cloud cold start)
    with st.spinner("Loading faculty index… (first launch may take a few minutes)"):
        ok, msg = _ensure_index_built()
    if not ok:
        st.error(f"Index error: {msg}")
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            str(Path(__file__).parent / "BSPH_logo_rgb_horizontal_blue.png"),
            use_container_width=True,
        )
        st.markdown("## 🔬 ResearchDating")
        st.markdown(
            "**ResearchDating** matches incoming master's students at the "
            "Johns Hopkins Bloomberg School of Public Health with faculty "
            "research mentors whose work aligns with their interests.\n\n"
            "Enter your research interests and the tool will return 3–5 ranked "
            "faculty matches, each with a plain-language explanation and a "
            "personalized draft outreach email."
        )

        st.markdown("---")
        st.markdown("**How to use**")
        st.markdown(
            "1. *(Optional)* Filter by department\n"
            "2. Describe your research interests in the text box\n"
            "3. Click **Find Matching Faculty**\n"
            "4. Review matches and expand the draft email for each"
        )

        st.markdown("---")
        st.markdown(
            "**Learn more** — [GitHub repository](https://github.com/hbmehta/genai-class-project)"
        )
        st.markdown(
            "<small>Hopkins MBA GenAI course project · BU.330.760 · Spring 2026</small>",
            unsafe_allow_html=True,
        )

    # ── Main area ─────────────────────────────────────────────────────────────
    st.title("ResearchDating")
    st.caption("Find faculty research mentors at Johns Hopkins Bloomberg School of Public Health")
    st.info(_DISCLAIMER)
    st.divider()

    # Clarification follow-up state
    if st.session_state["clarification_pending"]:
        st.markdown(f"**{st.session_state['follow_up_question']}**")
        refined = st.text_input("Your answer:", key="refined_input")

        col1, col2 = st.columns([1, 5])
        with col1:
            search_btn = st.button("Search", type="primary")
        with col2:
            if st.button("Start over"):
                for key in ["clarification_pending", "follow_up_question", "original_query", "results"]:
                    st.session_state[key] = False if key == "clarification_pending" else None
                st.rerun()

        if search_btn and refined.strip():
            combined_query = f"{st.session_state['original_query']} {refined}".strip()
            st.session_state["clarification_pending"] = False
            _run_search(combined_query, None)
        return

    # Main search form
    dept_choice = st.selectbox(
        "Filter by department (optional)",
        _DEPARTMENTS,
        index=0,
    )
    dept_filter = None if dept_choice == "All departments" else dept_choice

    query = st.text_area(
        "Describe your research interests",
        height=130,
        placeholder=_PLACEHOLDER,
        key="main_query",
    )

    if st.button("Find Matching Faculty", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please describe your research interests before searching.")
        else:
            st.session_state["original_query"] = query.strip()
            _run_search(query.strip(), dept_filter)

    # Render cached results
    if st.session_state["results"] is not None:
        result = st.session_state["results"]
        _render_results(
            result.get("matches") or [],
            result.get("low_confidence_warning", False),
            result.get("max_similarity", 0.0),
        )


def _run_search(query: str, dept_filter: str | None):
    with st.spinner("Searching faculty profiles…"):
        try:
            result = run_pipeline(query, department_filter=dept_filter)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return

    if result["status"] == "clarification_needed":
        st.session_state["clarification_pending"] = True
        st.session_state["follow_up_question"] = result["follow_up_question"]
        st.session_state["results"] = None
        st.rerun()
    else:
        st.session_state["results"] = result
        st.rerun()


if __name__ == "__main__":
    main()
