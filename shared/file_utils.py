"""
Utility helpers for processing file attachments uploaded via the Streamlit UI.

Supported attachment types:
  - PDF  → extracted plain text via pypdf
  - TXT  → raw decoded text
  - Image (PNG / JPG / JPEG / WEBP / GIF) → base64-encoded data URI for vision LLMs
"""
from __future__ import annotations

import base64
import io
from typing import Tuple

# ── PDF extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes, max_chars: int = 6000) -> str:
    """
    Extract plain text from PDF bytes using pypdf.
    Truncates to *max_chars* to stay within LLM context limits.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text.strip())

    full_text = "\n\n".join(filter(None, pages_text))
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + f"\n\n… [truncated at {max_chars} chars]"
    return full_text


# ── Plain-text extraction ───────────────────────────────────────────────────

def extract_text_from_txt(file_bytes: bytes, max_chars: int = 6000) -> str:
    """Decode a plain-text file, truncating to *max_chars*."""
    text = file_bytes.decode("utf-8", errors="replace").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n… [truncated at {max_chars} chars]"
    return text


# ── Image → base64 data URI ─────────────────────────────────────────────────

_MIME_MAP = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}

SUPPORTED_IMAGE_EXTS = set(_MIME_MAP.keys())
SUPPORTED_DOC_EXTS = {"pdf", "txt"}
SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_DOC_EXTS


def image_to_base64_uri(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Convert raw image bytes to a base64 data URI.

    Returns:
        (data_uri, mime_type)  e.g. ("data:image/png;base64,…", "image/png")
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = _MIME_MAP.get(ext, "image/png")
    b64 = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}", mime


def is_image(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in SUPPORTED_IMAGE_EXTS
