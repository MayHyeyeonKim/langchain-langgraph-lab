"""Small, explicit document adapters used before an LLM sees any content."""

from pathlib import Path

from pypdf import PdfReader


class DocumentExtractionError(ValueError):
    """Raised when a submitted document cannot provide reviewable text."""


def normalize_text(text: str) -> str:
    """Remove noisy whitespace while preserving paragraph boundaries."""
    paragraphs = [" ".join(part.split()) for part in text.splitlines()]
    return "\n".join(part for part in paragraphs if part).strip()


def extract_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    text = normalize_text("\n".join(page.extract_text() or "" for page in reader.pages))
    if not text:
        raise DocumentExtractionError(
            "The PDF contains no extractable text. Scanned PDFs require OCR, which is not enabled yet."
        )
    return text


def read_document(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() == ".pdf":
        return extract_pdf_text(file_path)
    return normalize_text(file_path.read_text(encoding="utf-8"))
