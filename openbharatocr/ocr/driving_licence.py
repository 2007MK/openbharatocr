"""
Driving licence OCR extractor.

Uses Tesseract (pytesseract) for OCR.
"""

import logging
import re

from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


def _extract_text(image_path: str) -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img).strip()


def extract_driving_licence_number(text: str) -> str:
    """Extract driving licence number from OCR text."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    pattern = r"\b[A-Z]{2}[- ]?\d{2}[- ]?\d{11}\b|\b[A-Z]{2}\d{2}\s\d{11}\b"
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def extract_all_dates(text: str):
    """
    Extract all dates and classify as DOB / issue dates / validity.

    Returns (dob, doi_list, validity_list).
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    matches = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text)
    dob = matches[0] if matches else None
    doi = matches[1:-1] if len(matches) > 1 else []
    validity = [matches[-1]] if len(matches) > 1 else []
    return dob, doi, validity


def clean_input(matches: list) -> list:
    """Split name tokens on newlines."""
    if not isinstance(matches, list):
        raise TypeError("Input must be a list")
    cleaned = []
    for m in matches:
        for part in str(m).split("\n"):
            if part.strip():
                cleaned.append(part.strip())
    return cleaned


def extract_all_names(text: str) -> str:
    """Extract a name from OCR text (first two words after removing stopwords)."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    stopwords = {"INDIA", "TRANSPORT", "LICENCE"}
    text = re.sub(r"(?i)\bname[:\-]?", "", text).strip()
    words = text.split()
    if any(sw in words for sw in stopwords):
        return ""
    return " ".join(words[:2]) if len(words) >= 2 else ""


def extract_address_regex(text: str) -> str:
    """Extract address using regex."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    m = re.search(r"Address\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).split("\n\n")[0].strip()
    m = re.search(r"ADDRESS\s*-\s*(.*)", text, flags=re.IGNORECASE)
    if m:
        return re.split(r"\d{6}|Other", m.group(1))[0].strip()
    return ""


def extract_address(image_path: str) -> str:
    """Extract address from an image using OCR."""
    try:
        return extract_address_regex(_extract_text(image_path))
    except FileNotFoundError:
        raise


def extract_auth_allowed(text: str) -> list:
    """Extract vehicle authorization codes."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    return re.findall(r"(MCWG|LMV|TRANS|M\.CYL\.)", text)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def driving_licence(image_path: str) -> dict:
    """Extract all details from a driving licence image."""
    text = _extract_text(image_path)
    dob, doi, validity = extract_all_dates(text)
    return {
        "licence_number": extract_driving_licence_number(text),
        "dates": {"dob": dob, "doi": doi, "validity": validity},
        "name": extract_all_names(text),
        "address": extract_address_regex(text),
        "auth_types": extract_auth_allowed(text),
        "raw_text": text,
    }
