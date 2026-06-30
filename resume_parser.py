"""
Resume text extraction. Supports .pdf, .docx, .txt.
Kept deliberately simple — just gets plain text out for the LLM to read.
No layout/section parsing; the LLM handles interpreting the content.
"""

from pathlib import Path

import pdfplumber
from docx import Document


def extract_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts).strip()

    if suffix == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()

    if suffix == ".txt":
        return Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()

    raise ValueError(f"Unsupported resume file type: {suffix}. Use .pdf, .docx, or .txt.")
