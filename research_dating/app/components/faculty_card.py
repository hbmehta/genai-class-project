"""Render a single faculty match card in Streamlit."""

import streamlit as st


def render_faculty_card(match: dict, index: int) -> None:
    """
    Render one faculty result as a styled Streamlit container.

    Displays: rank badge, name, department, explanation, overlap term tags,
    lab page link, and expandable draft outreach email.
    """
    with st.container(border=True):
        col_rank, col_info = st.columns([1, 11])

        with col_rank:
            st.markdown(
                f"<div style='background:#1f4e79;color:white;border-radius:50%;"
                f"width:40px;height:40px;display:flex;align-items:center;"
                f"justify-content:center;font-weight:bold;font-size:1.1em;'>"
                f"#{match['rank']}</div>",
                unsafe_allow_html=True,
            )

        with col_info:
            dept = match.get("department", "")
            st.markdown(f"**{match['faculty_name']}** &nbsp;·&nbsp; *{dept}*", unsafe_allow_html=True)
            st.write(match.get("explanation", ""))

            # Overlap term chips
            terms = match.get("key_overlap_terms", [])
            if terms:
                tag_html = " ".join(
                    f"<span style='background:#e8f4fd;padding:3px 10px;border-radius:12px;"
                    f"font-size:0.82em;color:#1f4e79;margin-right:4px'>{t}</span>"
                    for t in terms
                )
                st.markdown(tag_html, unsafe_allow_html=True)
                st.write("")  # spacing

            # Lab page link
            lab_url = match.get("lab_url", "")
            if lab_url:
                st.link_button("Visit Lab Page →", lab_url)

            # Draft email expander
            email_draft = match.get("email_draft", "")
            if email_draft:
                with st.expander("Draft Outreach Email"):
                    st.caption(
                        "Copy and personalize before sending. "
                        "Verify all details independently before reaching out."
                    )
                    st.text_area(
                        label="Email draft",
                        value=email_draft,
                        height=220,
                        key=f"email_draft_{index}",
                        label_visibility="collapsed",
                    )
