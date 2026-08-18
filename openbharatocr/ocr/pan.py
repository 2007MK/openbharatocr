"""
PAN Card OCR extractor.

Uses the shared PaddleOCR engine.  No external network calls; all
processing is local.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from openbharatocr.core.engine import extract_text_with_coords_paddle

logger = logging.getLogger(__name__)


class PANCardExtractor:
    """Extract fields from a PAN card image."""

    def __init__(self):
        # PAN number format: 5 letters, 4 digits, 1 letter
        self.pan_pattern = r"[A-Z]{5}[0-9]{4}[A-Z]{1}"

        self.date_patterns = [
            r"\d{2}/\d{2}/\d{4}",
            r"\d{2}-\d{2}-\d{4}",
            r"\d{2}\.\d{2}\.\d{4}",
        ]

        self.name_titles = [
            "श्री", "श्रीमती", "कुमारी",
            "shri", "smt", "kumari",
            "mr", "mrs", "ms", "dr", "sh",
        ]

        self.exclude_words = [
            # Long tokens (substring match is safe — never part of a real name)
            "government", "india", "income", "tax", "department",
            "permanent", "account", "number", "signature", "photo",
            "card", "pan", "specimen", "copy", "original",
            "भारत", "सरकार", "आयकर", "विभाग", "स्थायी",
            "खाता", "संख्या", "हस्ताक्षर", "फोटो", "प्रति", "मूल",
            "govt", "deartment", "govtofindia", "incometax",
            "pemanentaoun", "nambercard", "danotbth",
            "bitor", "fenhтst",
            # OCR misreads of 'GOVT OF INDIA' header (long enough for safe substring)
            "covt", "indla", "govl", "gnvt", "iindia",
        ]

        # Short tokens that must only match as whole words (use word-boundary
        # regex in is_valid_name; listing them here for doc purposes only).
        # DO NOT put them in self.exclude_words — doing so would reject any
        # name that happens to *contain* that sequence (e.g. 'ee' in TWITTERPREET).
        self._short_word_exclusions = [
            r"\bof\b", r"\bhra\b", r"\brr\b",
            r"\bee\b", r"\benrsh\b", r"\bfomse\b",
        ]

        self.ignore_text_patterns = [
            r"income.*tax.*department",
            r"govt.*of.*india",
            # OCR-corrupted variants of 'GOVT OF INDIA'
            r"covt.{0,6}(of|or|0f).{0,6}ind",
            r"gov[lt].{0,6}(of|or).{0,6}ind",
            r"permanent.*account.*number",
            r"signature",
            r"date.*of.*birth",
            r"^[A-Z]{5}[0-9]{4}[A-Z]$",
            r"^\d{2}/\d{2}/\d{4}$",
            r"^[0-9]+$",
            r"^[A-Z]$",
            r"^[A-Z]{1,3}$",
            r"\bcard\b",
            r"\bnumber\b",
            r"\bpermanent\b",
            r"\baccount\b",
            r"\bspecimen\b",
            r"\bhra\b",
            r"\brr\b",
            r"\bbitor\b",
            r"\bee\b",
            r"\benrsh\b",
            r"\bfomse\b",
        ]

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def preprocess_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width = img_rgb.shape[:2]
        if width < 800:
            scale = 800 / width
            img_rgb = cv2.resize(
                img_rgb,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text)
        cleaned = re.sub(r"[^\w\s/.-]", "", cleaned)
        return cleaned.strip()

    def clean_name(self, name: str) -> str:
        if not name:
            return ""
        words = name.strip().split()
        cleaned_words = [
            w for w in words
            if w.lower() not in self.name_titles
            and len(w) > 1
            and w.lower() not in ["sh", "smt", "shri"]
        ]
        if not cleaned_words:
            return ""
        result = " ".join(cleaned_words)
        result = re.sub(r"[^\w\s]", "", result)
        return result.title().strip()

    def normalize_for_matching(self, text: str) -> str:
        text = text.upper()
        text = text.replace("'", "")
        text = re.sub(r"[^A-Z\u0900-\u097F]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ------------------------------------------------------------------
    # Name validity
    # ------------------------------------------------------------------

    def is_valid_name(self, text: str, min_confidence: float = 0.6) -> bool:
        if not text or len(text) < 3:
            return False

        cleaned = self.clean_text(text).lower()

        # Regex patterns: word-boundary aware
        for pattern in self.ignore_text_patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                logger.debug("Rejected '%s' (ignore pattern: %s)", text, pattern)
                return False

        # Long exclude words: substring containment is safe (they never appear
        # inside real person-name syllables).
        for word in self.exclude_words:
            if word in cleaned:
                logger.debug("Rejected '%s' (exclude word: %s)", text, word)
                return False

        # Short word-only exclusions: must match as a whole word, not as a
        # substring, so that e.g. 'ee' rejects the token 'EE' but not
        # 'TWITTERPREET' or 'MANPREET'.
        for pattern in self._short_word_exclusions:
            if re.search(pattern, cleaned):
                logger.debug("Rejected '%s' (short-word exclusion: %s)", text, pattern)
                return False

        if re.search(r"\d", cleaned):
            return False

        words = cleaned.split()
        if len(words) < 2 and len(cleaned) < 6:
            return False

        for word in words:
            if not word[0].isalpha() or len(word) < 2:
                return False

        vowels = sum(1 for c in cleaned if c in "aeiou")
        consonants = sum(1 for c in cleaned if c.isalpha() and c not in "aeiou")
        if vowels == 0 or consonants == 0:
            return False

        if len(cleaned) > 4:
            alpha_chars = [c for c in cleaned if c.isalpha()]
            if alpha_chars:
                vowel_ratio = vowels / len(alpha_chars)
                if vowel_ratio < 0.1 or vowel_ratio > 0.8:
                    return False

        logger.debug("Accepted '%s' as valid name", text)
        return True

    def is_likely_name(self, text: str) -> bool:
        return self.is_valid_name(text)

    # ------------------------------------------------------------------
    # Field-label detectors
    # ------------------------------------------------------------------

    def is_father_label(self, text: str) -> bool:
        normalized = self.normalize_for_matching(text)
        compact = normalized.replace(" ", "")
        father_patterns = [
            "FATHER", "FATHERS NAME", "FATHER NAME",
            "PITA", "PITA KA NAAM", "PITAJI",
        ]
        return any(
            p in normalized or p.replace(" ", "") in compact
            for p in father_patterns
        )

    def is_name_label(self, text: str) -> bool:
        normalized = self.normalize_for_matching(text)
        compact = normalized.replace(" ", "")

        if self.is_father_label(text):
            return False
        if "PERMANENT ACCOUNT NUMBER" in normalized:
            return False
        if "DATE OF BIRTH" in normalized:
            return False
        if "NAME" in normalized:
            return True
        if "ERANAME" in compact:
            return True
        return False

    def is_dob_label(self, text: str) -> bool:
        normalized = self.normalize_for_matching(text)
        compact = normalized.replace(" ", "")
        return (
            "DATE OF BIRTH" in normalized
            or "DATEOFBIRTH" in compact
            or "DOB" in compact
            or "BIRTH" in normalized
            or "JANM" in normalized
        )

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def _horizontal_overlap(self, a: Dict, b: Dict) -> float:
        try:
            ax = [p[0] for p in a["bbox"]]
            bx = [p[0] for p in b["bbox"]]
            a_left, a_right = min(ax), max(ax)
            b_left, b_right = min(bx), max(bx)
            overlap = max(0.0, min(a_right, b_right) - max(a_left, b_left))
            return overlap / max(1.0, min(a_right - a_left, b_right - b_left))
        except Exception:
            return 0.0

    def _find_value_near_label(
        self,
        label_item: Dict,
        text_data: List[Dict],
        value_type: str = "name",
    ) -> Optional[Dict]:
        label_y = label_item["center_y"]
        candidates = []

        blocked_words = [
            "NAME", "FATHER", "FATHERS", "DATE", "BIRTH",
            "SIGNATURE", "PERMANENT", "ACCOUNT", "NUMBER",
            "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA",
        ]

        for item in text_data:
            if item is label_item:
                continue
            text = item["text"].strip()
            if not text:
                continue

            if value_type == "name":
                normalized = self.normalize_for_matching(text)
                compact = normalized.replace(" ", "")
                if any(
                    w in normalized.split() or w in compact
                    for w in blocked_words
                ):
                    continue
                if not self.is_valid_name(text, min_confidence=0.65):
                    continue
                cleaned = self.clean_name(text)
                if not cleaned or len(cleaned) < 4:
                    continue

            elif value_type == "dob":
                cleaned = self.clean_text(text)
                if not any(re.search(p, cleaned) for p in self.date_patterns):
                    continue
            else:
                continue

            dy = item["center_y"] - label_y
            abs_dy = abs(dy)
            if abs_dy > 250:
                continue

            overlap = self._horizontal_overlap(label_item, item)
            center_distance = abs(item["center_x"] - label_item["center_x"])

            score = max(0.0, 1.0 - abs_dy / 250.0) * 35
            score += overlap * 35
            score += float(item["confidence"]) * 20

            label_normalized = self.normalize_for_matching(label_item["text"])
            is_father_lbl = "FATHER" in label_normalized or "FATHERS" in label_normalized

            if value_type == "name":
                if is_father_lbl:
                    score += 25 if dy > 0 else -20
                else:
                    score += 25 if dy < 0 else -15
            elif value_type == "dob":
                score += 15 if dy < 0 else -10

            if center_distance > 350:
                score -= 30
            elif center_distance > 200:
                score -= 10

            candidates.append({"score": score, "item": item, "dy": dy, "overlap": overlap})

        if not candidates:
            return None

        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        logger.debug(
            "Spatial candidate for '%s': '%s' (dy=%.1f, overlap=%.2f, score=%.1f)",
            label_item["text"], best["item"]["text"], best["dy"], best["overlap"], best["score"],
        )
        return best["item"]

    def _find_nearest_valid_below_label(
        self,
        label_item: Dict,
        text_data: List[Dict],
        blocked_words: List[str],
        max_dy: float = 200.0,
        min_overlap: float = 0.0,
    ) -> Optional[Dict]:
        """
        Return the spatially closest valid-name token that appears **directly
        below** *label_item* in reading order.

        Selection criteria (in priority order):
        1. Must pass ``is_valid_name`` (rejects document noise, labels, digits).
        2. Must have ``dy > 0`` (below the label) and ``dy <= max_dy``.
        3. Must not contain any of ``blocked_words``.
        4. Among survivors: prefer smallest ``dy``; use horizontal overlap as
           tiebreaker only when two items are within 15 pixels of each other
           vertically.

        This is used specifically for father's-name extraction, where the
        PAN card layout guarantees that the father's name is the *first*
        valid text line directly under the label — not a score-maximising
        candidate that happens to have high OCR confidence.
        """
        candidates = []

        for item in text_data:
            if item is label_item:
                continue
            text = item["text"].strip()
            if not text:
                continue

            # Must be a valid person name
            if not self.is_valid_name(text, min_confidence=0.65):
                continue
            cleaned = self.clean_name(text)
            if not cleaned or len(cleaned) < 4:
                continue

            # Must not contain label/document blocked words
            normalized = self.normalize_for_matching(text)
            compact = normalized.replace(" ", "")
            if any(w in normalized.split() or w in compact for w in blocked_words):
                continue

            # Must be below the label
            dy = item["center_y"] - label_item["center_y"]
            if dy <= 0 or dy > max_dy:
                continue

            overlap = self._horizontal_overlap(label_item, item)
            if overlap < min_overlap:
                continue

            candidates.append({"item": item, "dy": dy, "overlap": overlap})

        if not candidates:
            return None

        # Sort: primarily by vertical proximity, secondarily by horizontal overlap
        # Two items within 15 px of each other vertically are treated as tied.
        candidates.sort(key=lambda c: (round(c["dy"] / 15), -c["overlap"]))
        best = candidates[0]
        logger.debug(
            "Nearest-below candidate for '%s': '%s' (dy=%.1f, overlap=%.2f)",
            label_item["text"], best["item"]["text"], best["dy"], best["overlap"],
        )
        return best["item"]

    # ------------------------------------------------------------------
    # Name extraction
    # ------------------------------------------------------------------

    def extract_names_with_keywords(
        self, text_data: List[Dict]
    ) -> Tuple[Optional[str], Optional[str]]:
        name = None
        father_name = None

        sorted_data = sorted(text_data, key=lambda i: (i["center_y"], i["center_x"]))

        # 1. Father's Name via label
        father_label = next((i for i in sorted_data if self.is_father_label(i["text"])), None)

        if father_label is not None:
            # Use nearest-below strategy: the father's name is the first valid
            # text line directly under the label in reading order.  This is
            # more robust than a score-maximising approach when high-confidence
            # noise (e.g. OCR misread of the 'GOVT OF INDIA' header) appears
            # nearby with a slightly better score.
            _father_blocked = [
                "NAME", "FATHER", "FATHERS", "DATE", "BIRTH",
                "SIGNATURE", "PERMANENT", "ACCOUNT", "NUMBER",
                "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA",
            ]
            candidate = self._find_nearest_valid_below_label(
                father_label, sorted_data, _father_blocked
            )
            if candidate:
                father_name = self.clean_name(candidate["text"])
                logger.debug("Label-based father's name (nearest-below): %s", father_name)

        # 2. Applicant name — first valid name directly ABOVE father's label
        if father_label is not None:
            blocked_words = [
                "NAME", "FATHER", "FATHERS", "DATE", "BIRTH",
                "SIGNATURE", "PERMANENT", "ACCOUNT", "NUMBER",
                "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA",
            ]
            applicant_candidates = []

            for item in sorted_data:
                if item is father_label:
                    continue
                text = item["text"].strip()
                if not text:
                    continue
                if father_name:
                    if self.clean_name(text).lower() == father_name.lower():
                        continue

                normalized = self.normalize_for_matching(text)
                compact = normalized.replace(" ", "")
                if any(w in normalized.split() or w in compact for w in blocked_words):
                    continue
                if not self.is_valid_name(text, min_confidence=0.65):
                    continue

                dy = item["center_y"] - father_label["center_y"]
                if dy >= 0 or abs(dy) > 220:
                    continue

                overlap = self._horizontal_overlap(father_label, item)
                score = (
                    max(0.0, 1.0 - abs(dy) / 220.0) * 50
                    + overlap * 30
                    + float(item["confidence"]) * 20
                )
                applicant_candidates.append({"item": item, "score": score})

            if applicant_candidates:
                applicant_candidates.sort(key=lambda c: c["score"], reverse=True)
                name = self.clean_name(applicant_candidates[0]["item"]["text"])
                logger.debug("Applicant name (anchored above father's label): %s", name)

        # 3. Fallback: generic Name label
        if not name:
            for item in sorted_data:
                if not self.is_name_label(item["text"]):
                    continue
                normalized = self.normalize_for_matching(item["text"])
                compact = normalized.replace(" ", "")
                if "ERANAME" in compact or "FATHERS" in compact or "FATHER" in compact:
                    continue
                candidate = self._find_value_near_label(item, sorted_data, "name")
                if candidate:
                    candidate_name = self.clean_name(candidate["text"])
                    if father_name and candidate_name.lower() == father_name.lower():
                        continue
                    name = candidate_name
                    logger.debug("Fallback label-based applicant name: %s", name)
                    break

        return name, father_name

    def extract_names_positional(
        self, text_data: List[Dict]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Positional fallback when labels are not reliably detected."""
        blocked_words = [
            "NAME", "FATHER", "FATHERS", "DATE", "BIRTH",
            "SIGNATURE", "PERMANENT", "ACCOUNT", "NUMBER",
            "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA",
        ]

        valid_candidates = []
        logger.debug("=== Positional name extraction ===")

        for item in text_data:
            text = item["text"]
            normalized = self.normalize_for_matching(text)
            compact = normalized.replace(" ", "")

            if any(w in normalized.split() or w in compact for w in blocked_words):
                continue

            if (
                item["confidence"] >= 0.65
                and self.is_valid_name(text, min_confidence=0.65)
            ):
                cleaned = self.clean_name(text)
                if cleaned and len(cleaned) >= 4:
                    valid_candidates.append({
                        "name": cleaned,
                        "center_y": item["center_y"],
                        "center_x": item["center_x"],
                        "confidence": item["confidence"],
                    })

        # Deduplicate
        seen: set = set()
        unique = []
        for c in valid_candidates:
            key = c["name"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda c: c["center_y"])

        name = unique[0]["name"] if len(unique) >= 1 else None
        father_name = unique[1]["name"] if len(unique) >= 2 else None
        return name, father_name

    def find_names_improved(self, text_data: List[Dict]) -> Dict[str, str]:
        """
        Robust PAN name extraction.

        Priority:
        1. Label + spatial relationship
        2. Positional fallback
        3. Sanity check (name ≠ father's name)
        """
        sorted_data = sorted(text_data, key=lambda i: (i["center_y"], i["center_x"]))
        logger.debug("=== PAN name extraction ===")

        name, father_name = self.extract_names_with_keywords(sorted_data)

        if not name or not father_name:
            logger.debug("Label extraction incomplete; trying positional fallback.")
            pos_name, pos_father = self.extract_names_positional(sorted_data)
            if not name and pos_name:
                name = pos_name
            if not father_name and pos_father:
                if not name or pos_father.lower() != name.lower():
                    father_name = pos_father

        if name and father_name and name.lower() == father_name.lower():
            logger.debug("Applicant and father names identical; discarding father name.")
            father_name = ""

        return {"name": name or "", "father_name": father_name or ""}

    # ------------------------------------------------------------------
    # PAN number
    # ------------------------------------------------------------------

    def find_pan_number(self, text_data: List[Dict]) -> Optional[str]:
        for item in text_data:
            text = self.clean_text(item["text"])
            match = re.search(self.pan_pattern, text.upper())
            if match:
                return match.group()
            text_upper = text.upper().replace(" ", "")
            if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", text_upper):
                return text_upper
        return None

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    def find_dates(self, text_data: List[Dict]) -> List[str]:
        dates = []
        for item in text_data:
            text = self.clean_text(item["text"])
            for pattern in self.date_patterns:
                dates.extend(re.findall(pattern, text))
        return dates

    def find_date_of_birth(self, text_data: List[Dict]) -> Optional[str]:
        sorted_data = sorted(text_data, key=lambda i: (i["center_y"], i["center_x"]))

        # Label-based
        for item in sorted_data:
            if not self.is_dob_label(item["text"]):
                continue
            candidate = self._find_value_near_label(item, sorted_data, "dob")
            if candidate:
                cleaned = self.clean_text(candidate["text"])
                for pattern in self.date_patterns:
                    match = re.search(pattern, cleaned)
                    if match and self.validate_date(match.group()):
                        logger.debug("Label-based DOB: %s", match.group())
                        return match.group()

        # Fallback
        valid_dates = [d for d in self.find_dates(sorted_data) if self.validate_date(d)]
        if valid_dates:
            return valid_dates[0]
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_pan(self, pan: str) -> bool:
        if not pan:
            return False
        return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan.upper().replace(" ", "")))

    def validate_date(self, date_str: str) -> bool:
        for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                age = datetime.now().year - date_obj.year
                return 1 <= age <= 120
            except ValueError:
                continue
        return False

    # ------------------------------------------------------------------
    # Public extraction
    # ------------------------------------------------------------------

    def extract_pan_details(self, image_path: str) -> Dict:
        """Extract PAN card fields from *image_path*."""
        import tempfile
        import os
        try:
            processed_image = self.preprocess_image(image_path)
            text_data = extract_text_with_coords_paddle(processed_image)
            
            if not text_data:
                return {"error": "No text could be extracted from the image"}

            pan_number = self.find_pan_number(text_data)
            names = self.find_names_improved(text_data)
            date_of_birth = self.find_date_of_birth(text_data)

            result: Dict = {
                "pan_number": pan_number if self.validate_pan(pan_number) else None,
                "name": names.get("name", ""),
                "father_name": names.get("father_name", ""),
                "date_of_birth": date_of_birth,
                "raw_text": [item["text"] for item in text_data],
            }

            # Field-level OCR confidence
            field_confidence = {
                "pan_number": 0.0,
                "name": 0.0,
                "father_name": 0.0,
                "date_of_birth": 0.0,
            }

            if result["pan_number"]:
                norm_pan = result["pan_number"].upper().replace(" ", "")
                for item in text_data:
                    if self.clean_text(item["text"]).upper().replace(" ", "") == norm_pan:
                        field_confidence["pan_number"] = float(item["confidence"])
                        break

            if result["name"]:
                for item in text_data:
                    if self.clean_name(item["text"]).lower() == result["name"].lower():
                        field_confidence["name"] = float(item["confidence"])
                        break

            if result["father_name"]:
                for item in text_data:
                    if self.clean_name(item["text"]).lower() == result["father_name"].lower():
                        field_confidence["father_name"] = float(item["confidence"])
                        break

            if result["date_of_birth"]:
                for item in text_data:
                    if result["date_of_birth"] in self.clean_text(item["text"]):
                        field_confidence["date_of_birth"] = float(item["confidence"])
                        break

            result["field_confidence"] = field_confidence

            # Completeness score (NOT accuracy)
            score = 0
            if result["pan_number"]:
                score += 40
            if result["name"]:
                score += 30
            if result["father_name"]:
                score += 20
            if result["date_of_birth"]:
                score += 10

            result["confidence_score"] = score
            result["confidence_score_type"] = "completeness"
            result["extraction_confidence"] = (
                "high" if score >= 90 else "medium" if score >= 60 else "low"
            )

            return result

        except Exception as exc:
            logger.exception("Error processing PAN image: %s", image_path)
            return {"error": f"Error processing image: {exc}"}