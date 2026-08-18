"""
OpenBharatOCR — Local OCR library for Indian government documents.

Quick start::

    import openbharatocr

    result = openbharatocr.pan("pan.jpg")
    result = openbharatocr.aadhaar("aadhaar-front.jpg")
    result = openbharatocr.passport("passport-front.jpg")
    result = openbharatocr.passport("passport-front.jpg", "passport-back.jpg")

All processing is local.  No data leaves the machine.
"""

__version__ = "0.5.0"
__author__ = "Kunal Kumar Kushwaha"
__email__ = "kunal@essentia.dev"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())


# ------------------------------------------------------------------
# Public convenience functions
# ------------------------------------------------------------------

def pan(image_path: str) -> dict:
    """
    Extract PAN card information from *image_path*.

    Returns a dict with keys:
        pan_number, name, father_name, date_of_birth,
        extraction_confidence, confidence_score, field_confidence, raw_text
    """
    from openbharatocr.ocr.pan import PANCardExtractor
    return PANCardExtractor().extract_pan_details(image_path)


def aadhaar(
    front_image_path: str,
    back_image_path: str = None,
) -> dict:
    """
    Extract Aadhaar card information.

    Pass only *front_image_path* for the front side.
    Pass both paths for a combined front+back result.
    """
    from openbharatocr.ocr.aadhaar import AadhaarOCR
    return AadhaarOCR().extract_aadhaar_details(front_image_path, back_image_path)


def passport(
    front_image_path: str,
    back_image_path: str = None,
) -> dict:
    """
    Extract passport information.

    Pass only *front_image_path* for the front page.
    Pass both paths for a combined front+back result.
    """
    from openbharatocr.ocr.passport import passport as _passport
    return _passport(front_image_path, back_image_path)


# ------------------------------------------------------------------
# Optional: document-type-specific helpers (back-compat)
# ------------------------------------------------------------------

def front_aadhaar(image_path: str) -> dict:
    """Extract Aadhaar front-side fields."""
    from openbharatocr.ocr.aadhaar import AadhaarOCR
    return AadhaarOCR().extract_front_aadhaar_details(image_path)


def back_aadhaar(image_path: str) -> dict:
    """Extract Aadhaar back-side fields."""
    from openbharatocr.ocr.aadhaar import AadhaarOCR
    return AadhaarOCR().extract_back_aadhaar_details(image_path)


def driving_licence(image_path: str) -> dict:
    """Extract driving licence information."""
    from openbharatocr.ocr.driving_licence import driving_licence as _dl
    return _dl(image_path)


def voter_id_front(image_path: str) -> dict:
    """Extract voter ID front-side information."""
    from openbharatocr.ocr.voter_id import voter_id_front as _vf
    return _vf(image_path)


def voter_id_back(image_path: str) -> dict:
    """Extract voter ID back-side information."""
    from openbharatocr.ocr.voter_id import voter_id_back as _vb
    return _vb(image_path)


def vehicle_registration(image_path: str) -> dict:
    """Extract vehicle registration certificate information."""
    from openbharatocr.ocr.vehicle_registration import vehicle_registration as _vr
    return _vr(image_path)


def water_bill(image_path: str) -> dict:
    """Extract water bill information."""
    from openbharatocr.ocr.water_bill import water_bill as _wb
    return _wb(image_path)


def birth_certificate(image_path: str) -> dict:
    """Extract birth certificate information."""
    from openbharatocr.ocr.birth_certificate import birth_certificate as _bc
    return _bc(image_path)


def degree(image_path: str) -> dict:
    """Extract degree certificate information."""
    from openbharatocr.ocr.degree import degree as _deg
    return _deg(image_path)


def flight_ticket(pdf_path: str) -> dict:
    """Extract flight ticket information from a PDF."""
    from openbharatocr.ocr.flight_ticket import ticket
    return ticket(pdf_path)


# ------------------------------------------------------------------
# Advanced API: OCRClient (optional, provides an object-oriented surface)
# ------------------------------------------------------------------

class OCRClient:
    """
    Optional object-oriented API.

    Example::

        client = OCRClient()
        result = client.pan("pan.jpg")
        result = client.passport("front.jpg", "back.jpg")
    """

    def pan(self, image_path: str) -> dict:
        return pan(image_path)

    def aadhaar(self, front_image_path: str, back_image_path: str = None) -> dict:
        return aadhaar(front_image_path, back_image_path)

    def passport(self, front_image_path: str, back_image_path: str = None) -> dict:
        return passport(front_image_path, back_image_path)

    def driving_licence(self, image_path: str) -> dict:
        return driving_licence(image_path)

    def voter_id_front(self, image_path: str) -> dict:
        return voter_id_front(image_path)

    def voter_id_back(self, image_path: str) -> dict:
        return voter_id_back(image_path)

    def vehicle_registration(self, image_path: str) -> dict:
        return vehicle_registration(image_path)

    def water_bill(self, image_path: str) -> dict:
        return water_bill(image_path)

    def birth_certificate(self, image_path: str) -> dict:
        return birth_certificate(image_path)

    def degree(self, image_path: str) -> dict:
        return degree(image_path)

    def flight_ticket(self, pdf_path: str) -> dict:
        return flight_ticket(pdf_path)

# ------------------------------------------------------------------
# __all__
# ------------------------------------------------------------------

__all__ = [
    # Simple functions
    "pan",
    "aadhaar",
    "passport",
    "flight_ticket",
    "front_aadhaar",
    "back_aadhaar",
    "driving_licence",
    "voter_id_front",
    "voter_id_back",
    "vehicle_registration",
    "water_bill",
    "birth_certificate",
    "degree",
    # Advanced
    "OCRClient",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
