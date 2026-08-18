"""
Aadhaar card OCR extractor.

Uses the shared PaddleOCR engine.  No external network calls; all
processing is local.
"""

import logging
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from openbharatocr.core.engine import extract_text_paddle

logger = logging.getLogger(__name__)


class AadhaarOCR:
    """Extract fields from Aadhaar card images (front and/or back)."""

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.medianBlur(enhanced, 3)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        struct_el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, struct_el)
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    def _crop(self, image: np.ndarray, region: str) -> np.ndarray:
        h, w = image.shape[:2]
        if region == "back_top_left":
            return image[0: int(h * 0.6), 0: int(w * 0.65)]
        if region == "front_details":
            return image[int(h * 0.3): h, 0:w]
        if region == "aadhaar_number":
            return image[int(h * 0.7): h, 0:w]
        return image

    # ------------------------------------------------------------------
    # Field extraction helpers
    # ------------------------------------------------------------------

    def _extract_name(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        garbage = {
            "government", "of", "india", "unique", "identification",
            "authority", "dob", "date of birth", "gender", "male",
            "female", "aadhaar", "address", "pin", "code", "www",
            "uidai", "mera", "card", "number", "id", "year", "birth",
            "vid", "virtual", "son", "daughter", "wife",
            "s/o", "d/o", "w/o", "father",
        }
        name_candidate = ""

        for i, line in enumerate(lines):
            lower = line.lower()
            if "government of india" in lower or "unique identification authority of india" in lower:
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if (
                        re.match(r"^[A-Za-z\s\.\-]+$", nxt)
                        and len(nxt.split()) >= 2
                        and 5 <= len(nxt) <= 50
                        and not any(kw in nxt.lower() for kw in garbage)
                    ):
                        return nxt.title()
            elif "name" in lower and ":" in line and i + 1 < len(lines):
                match = re.search(r"name[:\s]*([A-Z][a-zA-Z\s\.\-]{2,50})", line, re.I)
                if match:
                    candidate = match.group(1).strip()
                    if (
                        re.match(r"^[A-Za-z\s\.\-]+$", candidate)
                        and len(candidate.split()) >= 2
                        and 5 <= len(candidate) <= 50
                        and not any(kw in candidate.lower() for kw in garbage)
                    ):
                        return candidate.title()
                nxt = lines[i + 1].strip()
                if (
                    re.match(r"^[A-Za-z\s\.\-]+$", nxt)
                    and len(nxt.split()) >= 2
                    and 5 <= len(nxt) <= 50
                    and not any(kw in nxt.lower() for kw in garbage)
                ):
                    return nxt.title()

        for line in lines:
            clean = re.sub(r"[^\w\s]", " ", line).strip()
            lower = clean.lower()
            if (
                re.match(r"^[A-Za-z\s]+$", clean)
                and 2 <= len(clean.split()) <= 5
                and 5 <= len(clean) <= 50
                and not any(kw in lower for kw in garbage)
                and not (clean.isupper() and len(clean.split()) > 3)
            ):
                if re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z]+)+$", clean) and len(clean) >= 8:
                    return clean.title()
                if not name_candidate or (
                    len(clean) > len(name_candidate)
                    and "male" not in lower
                    and "female" not in lower
                ):
                    name_candidate = clean.title()

        return name_candidate

    def _extract_dob(self, text: str) -> str:
        current_year = datetime.now().year
        patterns = [
            r"(?:DOB|Date of Birth|D\.O\.B|D O B|Year of Birth)[:\s]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{4})\b",
            r"\b([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{4})\b",
            r"(?:DOB|Date of Birth|D\.O\.B|D O B|Year of Birth)[:\s]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2})\b",
            r"\b([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                dob = re.sub(r"[-]", "/", match.group(1))
                parts = dob.split("/")
                if len(parts) == 3:
                    try:
                        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                        if 0 < month <= 12 and 0 < day <= 31:
                            if len(parts[2]) == 2:
                                year = (2000 + year) if year <= (current_year % 100) else (1900 + year)
                            if 1900 <= year <= current_year:
                                return dob
                    except ValueError:
                        pass
        return ""

    def _extract_gender(self, text: str) -> str:
        norm = text.replace("T/FEMALE", "FEMALE").replace("M/MALE", "MALE").upper()
        if re.search(r"\bFEMALE\b", norm):
            return "Female"
        if re.search(r"\bMALE\b", norm):
            return "Male"
        if re.search(r"\bTRANSGENDER\b", norm):
            return "Transgender"
        match = re.search(r"\b(F|M|T)\b", norm)
        if match:
            return {"M": "Male", "F": "Female", "T": "Transgender"}.get(match.group(1), "")
        return ""

    def _extract_aadhaar_number(self, text: str) -> str:
        text_clean = re.sub(r"\s+", " ", text.replace("\n", " "))
        for pattern in [
            r"\b(\d{4}\s*\d{4}\s*\d{4})\b",
            r"\b(\d{12})\b",
            r"(\d{4}[\s\-]\d{4}[\s\-]\d{4})",
        ]:
            for match in re.findall(pattern, text_clean):
                clean = re.sub(r"[\s\-]", "", match)
                if len(clean) == 12 and clean.isdigit():
                    return f"{clean[:4]} {clean[4:8]} {clean[8:]}"
        return ""

    def _extract_relative_name_and_address(
        self, text: str
    ) -> Tuple[str, str, str]:
        relative_name = ""
        relation_type = "Not Found"

        temp = text
        temp = re.sub(r"[Ss]\s*/\s*[oO]", "S/o", temp)
        temp = re.sub(r"[Dd]\s*/\s*[oO]", "D/o", temp)
        temp = re.sub(r"[Ww]\s*/\s*[oO]", "W/o", temp)

        bad_kws = ["dob", "gender", "aadhaar", "address", "pin", "india",
                   "private", "limited", "mobile", "tel", "email"]

        def _is_good_name(name: str) -> bool:
            return (
                len(name.split()) >= 2
                and len(name) <= 60
                and bool(re.match(r"^[A-Za-z\s\.\-]+$", name))
                and not any(k in name.lower() for k in bad_kws)
            )

        relation_patterns = {
            "Husband": r"(W/o|W/O|Wife of)[:\s]*([A-Z][a-zA-Z\s\.\-]{2,60})",
            "Father":  r"(S/o|S/O|Son of)[:\s]*([A-Z][a-zA-Z\s\.\-]{2,60})",
            "Daughter": r"(D/o|D/O|Daughter of)[:\s]*([A-Z][a-zA-Z\s\.\-]{2,60})",
        }

        matched_string = ""
        for rel_type, pattern in relation_patterns.items():
            m = re.search(pattern, temp, re.I)
            if m:
                candidate = m.group(2).strip()
                if _is_good_name(candidate):
                    relative_name = candidate.title()
                    relation_type = rel_type
                    matched_string = m.group(0)
                    break

        if not relative_name:
            m = re.search(r"(Father)[:\s]*([A-Z][a-zA-Z\s\.\-]{2,60})", temp, re.I)
            if m:
                candidate = m.group(2).strip()
                if _is_good_name(candidate):
                    relative_name = candidate.title()
                    relation_type = "Father"
                    matched_string = m.group(0)

        if not relative_name:
            for line in [l.strip() for l in temp.split("\n") if l.strip()]:
                m = re.search(
                    r"^([A-Z][a-zA-Z\s\.\-]{3,60})\s*[,.]?\s*"
                    r"(?:H\.NO\.|HOUSE|VILLAGE|APARTMENT|STREET|ROAD|SECTOR|PHASE)",
                    line, re.I,
                )
                if m:
                    candidate = m.group(1).strip()
                    if _is_good_name(candidate):
                        relative_name = candidate.title()
                        if not re.search(r"(W/o|W/O|D/o|D/O)", line, re.I):
                            relation_type = "Father"
                        matched_string = m.group(0)
                        break

        # Build address from cleaned text
        cleaned_for_address = text
        if relative_name and matched_string:
            escaped = re.escape(matched_string)
            cleaned_for_address = re.sub(
                escaped + r"[:,\s]*", " ", cleaned_for_address, flags=re.I
            ).strip()
            for variant in [relative_name.title(), relative_name.upper()]:
                cleaned_for_address = cleaned_for_address.replace(variant, "").strip()
            cleaned_for_address = re.sub(r"\n\s*\n", "\n", cleaned_for_address).strip()
            cleaned_for_address = re.sub(r"\s{2,}", " ", cleaned_for_address).strip()

        ignore_patterns = [
            r"mera\s+aadhaar", r"www\.", r"uidai", r"https?://",
            r"email", r"@", r"contact\s+us", r"government",
            r"unique.*identification", r"vid", r"date of birth",
            r"gender", r"male", r"female", r"aadhaar", r"number",
            r"paddles", r"ocr", r"\d{4}\s*\d{4}\s*\d{4}", r"tel",
            r"mobile", r"uid", r"virtual\s+id",
        ]
        address_start_kws = [
            r"\baddress\b", r"\bपता\b", r"\bH\.NO\.\b",
            r"\bHOUSE\b", r"\bFLAT\b", r"\bVILLAGE\b",
            r"\bCOLONY\b", r"\bSTREET\b", r"\bROAD\b", r"\bSECTOR\b",
        ]

        address_lines = []
        collecting = False

        for line in [l.strip() for l in cleaned_for_address.split("\n") if l.strip()]:
            lower = line.lower()
            if any(re.search(kw, lower) for kw in address_start_kws):
                collecting = True
            if collecting or re.search(r"\b\d{6}\b", line) or re.search(r"\bH\.NO\.\s*\-?\d+", line, re.I):
                is_garbage = any(re.search(p, lower) for p in ignore_patterns)
                has_alnum = len(re.findall(r"[A-Za-z0-9]", line)) >= len(line) * 0.3
                if not is_garbage and len(line) > 5 and has_alnum:
                    address_lines.append(line)
            if len(address_lines) >= 5:
                break

        final_address = ", ".join(address_lines)
        final_address = re.sub(r"(?:PIN\s*CODE|PINCODE)[:\s]*", "", final_address, flags=re.I)
        final_address = re.sub(r"[,]{2,}", ",", final_address)
        final_address = re.sub(r"\s{2,}", " ", final_address).strip()
        final_address = re.sub(r"^[\s,]+|[\s,]+$", "", final_address)
        final_address = ", ".join(filter(None, final_address.split(", ")))

        return relation_type, relative_name, final_address

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    def extract_front_aadhaar_details(self, image_path: str) -> Dict:
        """Extract fields from the front side of an Aadhaar card."""
        processed = self._preprocess(image_path)

        text_full = extract_text_paddle(processed)
        text_details = extract_text_paddle(self._crop(processed, "front_details"))
        text_number = extract_text_paddle(self._crop(processed, "aadhaar_number"))

        def _best(a: str, b: str) -> str:
            return a if len(a) >= len(b) else b

        return {
            "document_type": "aadhaar_front",
            "fields": {
                "name":           _best(self._extract_name(text_full),          self._extract_name(text_details)),
                "date_of_birth":  _best(self._extract_dob(text_full),           self._extract_dob(text_details)),
                "gender":         _best(self._extract_gender(text_full),        self._extract_gender(text_details)),
                "aadhaar_number": _best(self._extract_aadhaar_number(text_full), self._extract_aadhaar_number(text_number)),
            },
            "raw_text": text_full,
        }

    def extract_back_aadhaar_details(self, image_path: str) -> Dict:
        """Extract fields from the back side of an Aadhaar card."""
        processed = self._preprocess(image_path)

        text_full = extract_text_paddle(processed)
        text_roi = extract_text_paddle(self._crop(processed, "back_top_left"))

        def _best(a: str, b: str) -> str:
            return a if len(a) >= len(b) else b

        rel_type1, rel_name1, addr1 = self._extract_relative_name_and_address(text_full)
        rel_type2, rel_name2, addr2 = self._extract_relative_name_and_address(text_roi)

        pincode = ""
        for text in (text_full, text_roi):
            m = re.search(r"\b(\d{6})\b", text)
            if m:
                pincode = m.group(1)
                break

        return {
            "document_type": "aadhaar_back",
            "fields": {
                "relation_type":  _best(rel_type1, rel_type2),
                "relative_name":  _best(rel_name1, rel_name2),
                "address":        _best(addr1, addr2),
                "pincode":        pincode,
            },
            "raw_text": text_full,
        }

    def extract_aadhaar_details(
        self, front_image_path: str, back_image_path: Optional[str] = None
    ) -> Dict:
        """
        Extract Aadhaar details from front and optionally back image.

        Returns a combined result when *back_image_path* is provided.
        """
        front = self.extract_front_aadhaar_details(front_image_path)

        if back_image_path is None:
            return front

        back = self.extract_back_aadhaar_details(back_image_path)

        return {
            "document_type": "aadhaar",
            "fields": {**front["fields"], **back["fields"]},
            "raw_text_front": front["raw_text"],
            "raw_text_back": back["raw_text"],
        }
