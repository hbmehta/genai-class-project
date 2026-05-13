"""Parse faculty CV PDFs into structured section dicts."""

import re
from pathlib import Path

import pdfplumber


_HEADER_PATTERN = re.compile(
    r"^(?:"
    r"[A-Z][A-Z\s&,/()-]{3,}"  # ALL CAPS lines (min 4 chars)
    r"|.{3,}\n[-=]{3,}"         # lines followed by underlines
    r")$"
)

_KNOWN_SECTIONS = [
    "education", "positions", "appointments", "experience",
    "research interests", "research", "publications", "grants",
    "funding", "awards", "honors", "teaching", "service",
    "professional activities", "mentoring", "trainees",
    "selected publications", "peer-reviewed", "books", "chapters",
    "presentations", "abstracts", "skills", "languages",
]


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    if _HEADER_PATTERN.match(stripped):
        return True
    lower = stripped.lower().rstrip(":").rstrip()
    return any(lower == s or lower.startswith(s) for s in _KNOWN_SECTIONS)


def extract_cv_text(pdf_path: str | Path) -> dict[str, str] | str:
    """
    Extract CV text from a PDF and return a section dict if headers are found,
    or a raw string fallback for unstructured CVs.

    Returns:
        dict: {section_name: section_text} if sections detected
        str:  raw full text if no sections detected (triggers sliding window chunker)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"CV not found: {pdf_path}")

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                full_text += text + "\n"

    if not full_text.strip():
        return ""

    sections: dict[str, str] = {}
    current_section = "preamble"
    current_lines: list[str] = []

    for line in full_text.split("\n"):
        if _is_section_header(line):
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = line.strip().rstrip(":").title()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    # Remove empty sections and the preamble if it's just whitespace
    sections = {k: v for k, v in sections.items() if v.strip()}

    # Fall back to raw string if no meaningful sections were detected
    if len(sections) <= 1:
        return full_text

    return sections
