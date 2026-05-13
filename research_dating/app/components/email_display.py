"""Standalone expandable email display component (used independently if needed)."""

import streamlit as st


def render_email_display(email_text: str, key: str) -> None:
    """Render an expandable, editable email draft."""
    with st.expander("Draft Outreach Email"):
        st.caption(
            "Copy and personalize before sending. "
            "Verify all details independently before reaching out."
        )
        st.text_area(
            label="Email draft",
            value=email_text,
            height=220,
            key=key,
            label_visibility="collapsed",
        )
