import cv2
import numpy as np
import re
from datetime import datetime
from paddleocr import PaddleOCR
import json
from typing import Dict, List, Tuple, Optional


class PANCardExtractor:
    def __init__(self):
        # Set up PaddleOCR - using English for now but could add more languages later
        self.ocr = PaddleOCR(lang="en")

        # This regex should catch all valid PAN numbers (5 letters, 4 digits, 1 letter)
        self.pan_pattern = r"[A-Z]{5}[0-9]{4}[A-Z]{1}"

        # Common date formats we see on PAN cards - could expand this if needed
        self.date_patterns = [
            r"\d{2}/\d{2}/\d{4}",
            r"\d{2}-\d{2}-\d{4}",
            r"\d{2}\.\d{2}\.\d{4}",
        ]

        # Keywords that typically appear near different fields on PAN cards
        # Adding both English and Hindi terms since PAN cards have both
        self.field_keywords = {
            "name": ["name", "श्री", "श्रीमती", "कुमारी", "shri", "smt", "kumari"],
            "father_name": [
                "father",
                "पिता",
                "father's name",
                "fathers name",
                "father name",
                "पिता का नाम",
                "पिताजी",
            ],
            "dob": ["birth", "जन्म", "date of birth", "dob", "born", "जन्म तिथि"],
            "pan": ["permanent account number", "pan", "account number"],
        }

        # Common title prefixes that we want to filter out from names
        self.name_titles = [
            "श्री",
            "श्रीमती",
            "कुमारी",
            "shri",
            "smt",
            "kumari",
            "mr",
            "mrs",
            "ms",
            "dr",
            "sh",
        ]

        # Words to ignore when trying to extract names - these show up a lot on PAN cards
        # but are definitely not names. Added some common OCR mistakes I've seen too
        self.exclude_words = [
            "government",
            "india",
            "income",
            "tax",
            "department",
            "permanent",
            "account",
            "number",
            "signature",
            "photo",
            "card",
            "pan",
            "specimen",
            "copy",
            "original",
            "भारत",
            "सरकार",
            "आयकर",
            "विभाग",
            "स्थायी",
            "खाता",
            "संख्या",
            "हस्ताक्षर",
            "फोटो",
            "प्रति",
            "मूल",
            "govt",
            "of",
            "deartment",
            "govtofindia",
            "incometax",
            "pemanentaoun",
            "nambercard",
            "danotbth",
            "hra",
            "rr",
            "bitor",
            "fenhтst",
            "ee",
            "enrsh",
            "fomse",  # These are OCR errors I've encountered
        ]

        # Patterns that match typical Indian names - helps validate if something is actually a name
        self.indian_name_patterns = [
            r"^[A-Z][a-z]+ [A-Z][a-z]+$",  # Standard First Last format
            r"^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$",  # First Middle Last
            r"^[A-Z]+ [A-Z]+ [A-Z]+$",  # Sometimes names are in all caps
            r"^[A-Z][A-Z]+ [A-Z][A-Z]+$",  # Multiple capital letters
            r"^[A-Z][a-z]+[A-Z][a-z]+$",  # Sometimes first and last are combined
        ]

        # Regex patterns to catch text we definitely want to ignore
        # This helps filter out boilerplate text from PAN cards
        self.ignore_text_patterns = [
            r"income.*tax.*department",
            r"govt.*of.*india",
            r"permanent.*account.*number",
            r"signature",
            r"date.*of.*birth",
            r"^[A-Z]{5}[0-9]{4}[A-Z]$",  # This would be the PAN number itself
            r"^\d{2}/\d{2}/\d{4}$",  # Date patterns
            r"^[0-9]+$",  # Just numbers
            r"^[A-Z]$",  # Single letters (probably OCR artifacts)
            r"^[A-Z]{1,3}$",  # Short abbreviations
            r"card",
            r"number",
            r"permanent",
            r"account",
            r"specimen",
            r"hra\s+rr",  # Specific OCR error I keep seeing
            r"bitor",
            r"fenhтst",
            r"ee",
            r"enrsh",
            r"fomse",
        ]

    def preprocess_image(self, image_path: str) -> np.ndarray:
        # Load the image and do some basic error checking
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Convert to RGB since that's what most processing expects
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Scale up small images - OCR works better on larger images
        height, width = img_rgb.shape[:2]
        if width < 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_rgb = cv2.resize(
                img_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )

        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Enhance contrast using CLAHE - helps with poor quality scans
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Remove noise which can confuse OCR
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Sharpen the image to make text clearer
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Convert back to RGB for final processing
        final_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

        return final_img

    def extract_text_with_coordinates(self, image: np.ndarray) -> List[Tuple]:
        # Try to run OCR - different versions of PaddleOCR have different interfaces
        try:
            results = self.ocr.predict(image)
        except (AttributeError, TypeError):
            try:
                results = self.ocr.ocr(image)
            except:
                results = self.ocr(image)

        extracted_data = []

        # Handle the results - format can vary between PaddleOCR versions
        if results and isinstance(results, list) and len(results) > 0:
            result_dict = results[0]

            # This handles the newer PaddleX format
            if isinstance(result_dict, dict) and "rec_texts" in result_dict:
                texts = result_dict.get("rec_texts", [])
                scores = result_dict.get("rec_scores", [])
                polygons = result_dict.get("rec_polys", [])

                print("Extracted text from PaddleX format:")
                for i, text in enumerate(texts):
                    confidence = scores[i] if i < len(scores) else 0.8
                    bbox = polygons[i] if i < len(polygons) else []

                    print(f"  - {text} (confidence: {confidence:.2f})")

                    # Only keep text with decent confidence
                    if confidence > 0.3:
                        if len(bbox) > 0:
                            center_x = float(np.mean(bbox[:, 0]))
                            center_y = float(np.mean(bbox[:, 1]))
                        else:
                            center_x = center_y = 0

                        extracted_data.append(
                            {
                                "text": text.strip(),
                                "confidence": confidence,
                                "bbox": (
                                    bbox.tolist() if hasattr(bbox, "tolist") else bbox
                                ),
                                "center_x": center_x,
                                "center_y": center_y,
                            }
                        )

                return extracted_data

            # This handles the standard PaddleOCR format
            elif isinstance(result_dict, list):
                lines = result_dict
                for line in lines:
                    try:
                        if line and isinstance(line, (list, tuple)) and len(line) >= 2:
                            bbox = line[0]
                            text_info = line[1]

                            # Extract text and confidence
                            if (
                                isinstance(text_info, (list, tuple))
                                and len(text_info) >= 2
                            ):
                                text = text_info[0]
                                confidence = text_info[1]
                            else:
                                text = str(text_info)
                                confidence = 0.8  # Default confidence if not provided

                            # Only keep high enough confidence text
                            if confidence > 0.3:
                                center_x = sum([point[0] for point in bbox]) / len(bbox)
                                center_y = sum([point[1] for point in bbox]) / len(bbox)

                                extracted_data.append(
                                    {
                                        "text": text.strip(),
                                        "confidence": confidence,
                                        "bbox": bbox,
                                        "center_x": center_x,
                                        "center_y": center_y,
                                    }
                                )
                    except Exception as e:
                        print(f"Error processing line {line}: {e}")
                        continue

        return extracted_data

    def clean_text(self, text: str) -> str:
        # Basic text cleaning - normalize whitespace and remove weird characters
        cleaned = re.sub(r"\s+", " ", text)
        cleaned = re.sub(r"[^\w\s/.-]", "", cleaned)
        return cleaned.strip()

    def clean_name(self, name: str) -> str:
        # Clean up extracted names to make them look proper
        if not name:
            return ""

        name = name.strip()

        # Split into words and filter out titles/prefixes
        words = name.split()
        cleaned_words = []

        for word in words:
            # Skip common titles and single letters (usually OCR errors)
            if (
                word.lower() not in self.name_titles
                and len(word) > 1
                and word.lower() not in ["sh", "smt", "shri"]
            ):
                cleaned_words.append(word)

        if not cleaned_words:
            return ""

        cleaned_name = " ".join(cleaned_words)

        # Remove special characters but keep spaces
        cleaned_name = re.sub(r"[^\w\s]", "", cleaned_name)

        # Make it look like a proper name (Title Case)
        cleaned_name = cleaned_name.title()

        return cleaned_name.strip()

    def is_valid_name(self, text: str, min_confidence: float = 0.6) -> bool:
        # Check if some text actually looks like a real Indian name
        if not text or len(text) < 3:
            return False

        # Clean first then check
        cleaned = self.clean_text(text).lower()

        # Run through our ignore patterns first
        for pattern in self.ignore_text_patterns:
            if re.search(pattern, cleaned):
                print(f"  -> Rejected '{text}' due to ignore pattern: {pattern}")
                return False

        # Check against our exclude word list
        for exclude_word in self.exclude_words:
            if exclude_word in cleaned:
                print(f"  -> Rejected '{text}' due to exclude word: {exclude_word}")
                return False

        # Names shouldn't have numbers in them
        if re.search(r"\d", cleaned):
            print(f"  -> Rejected '{text}' due to containing digits")
            return False

        # Full names may contain multiple words, but a single long
        # OCR token can also be a legitimate name.
        words = cleaned.split()
        if len(words) < 2 and len(cleaned) < 6:
            print(f"  -> Rejected '{text}' due to insufficient length/words")
            return False

        # Each word should look like part of a name
        for word in words:
            if not word[0].isalpha() or len(word) < 2:
                print(f"  -> Rejected '{text}' due to invalid word: {word}")
                return False

        # Check if it has a reasonable mix of vowels and consonants
        # Real names usually have both
        vowels = sum(1 for char in cleaned if char in "aeiou")
        consonants = sum(
            1 for char in cleaned if char.isalpha() and char not in "aeiou"
        )

        if vowels == 0 or consonants == 0:
            print(f"  -> Rejected '{text}' due to lack of vowels/consonants")
            return False

        # Check vowel ratio - too many or too few vowels is suspicious
        if len(cleaned) > 4:
            vowel_ratio = vowels / len([c for c in cleaned if c.isalpha()])
            if vowel_ratio < 0.1 or vowel_ratio > 0.8:
                print(
                    f"  -> Rejected '{text}' due to unusual vowel ratio: {vowel_ratio}"
                )
                return False

        print(f"  -> Accepted '{text}' as valid name")
        return True

    def is_likely_name(self, text: str) -> bool:
        # Keeping this for compatibility with any old code
        return self.is_valid_name(text)

    def find_pan_number(self, text_data: List[Dict]) -> Optional[str]:
        # Look for PAN numbers in the extracted text
        for item in text_data:
            text = self.clean_text(item["text"])
            pan_match = re.search(self.pan_pattern, text.upper())
            if pan_match:
                return pan_match.group()

            # Sometimes OCR adds spaces in PAN numbers, so check without spaces too
            text_upper = text.upper().replace(" ", "")
            if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", text_upper):
                return text_upper

        return None

    def find_dates(self, text_data: List[Dict]) -> List[str]:
        # Extract all date-like strings from the text
        dates = []
        for item in text_data:
            text = self.clean_text(item["text"])
            for pattern in self.date_patterns:
                matches = re.findall(pattern, text)
                dates.extend(matches)
        return dates

    def normalize_for_matching(self, text: str) -> str:
        """Normalize OCR text for robust field-label matching."""
        text = text.upper()
        text = text.replace("'", "")
        text = re.sub(r"[^A-Z\u0900-\u097F]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def is_father_label(self, text: str) -> bool:
        normalized = self.normalize_for_matching(text)
        compact = normalized.replace(" ", "")

        father_patterns = [
            "FATHER",
            "FATHERS NAME",
            "FATHER NAME",
            "PITA",
            "PITA KA NAAM",
            "PITAJI",
        ]

        return any(
            pattern in normalized or pattern.replace(" ", "") in compact
            for pattern in father_patterns
        )

    def is_name_label(self, text: str) -> bool:
        normalized = self.normalize_for_matching(text)
        compact = normalized.replace(" ", "")

        # Father's Name must never be treated as the applicant Name label.
        if self.is_father_label(text):
            return False

        # Boilerplate fields containing "Name" are not applicant-name labels.
        if "PERMANENT ACCOUNT NUMBER" in normalized:
            return False

        if "DATE OF BIRTH" in normalized:
            return False

        if "NAME" in normalized:
            return True

        # Common OCR corruption seen in PAN samples, e.g. "EraName".
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

    def _horizontal_overlap(self, a: Dict, b: Dict) -> float:
        """Estimate horizontal overlap between two OCR bounding boxes."""
        try:
            ax = [point[0] for point in a["bbox"]]
            bx = [point[0] for point in b["bbox"]]

            a_left, a_right = min(ax), max(ax)
            b_left, b_right = min(bx), max(bx)

            overlap = max(
                0.0,
                min(a_right, b_right) - max(a_left, b_left),
            )

            a_width = max(1.0, a_right - a_left)
            b_width = max(1.0, b_right - b_left)

            return overlap / min(a_width, b_width)
        except Exception:
            return 0.0

    def _find_value_near_label(
        self,
        label_item: Dict,
        text_data: List[Dict],
        value_type: str = "name",
    ) -> Optional[Dict]:
        """
        Find a field value using the actual PAN-card layout.

        PAN layouts are not reliably represented by OCR result order.
        We therefore use field-specific spatial relationships:
          - applicant Name: value is usually above the label
          - Father's Name: value is usually below the label
          - DOB: value is usually above the label

        Both directions are still considered, but the expected
        direction gets a strong score bonus.
        """
        label_y = label_item["center_y"]
        candidates = []

        for item in text_data:
            if item is label_item:
                continue

            text = item["text"].strip()

            if not text:
                continue

            if value_type == "name":
                # Never allow labels/boilerplate to become values.
                normalized = self.normalize_for_matching(text)
                compact = normalized.replace(" ", "")

                blocked_words = [
                    "NAME",
                    "FATHER",
                    "FATHERS",
                    "DATE",
                    "BIRTH",
                    "SIGNATURE",
                    "PERMANENT",
                    "ACCOUNT",
                    "NUMBER",
                    "INCOME",
                    "TAX",
                    "DEPARTMENT",
                    "GOVT",
                    "GOVERNMENT",
                    "INDIA",
                ]

                if any(
                    word in normalized.split() or word in compact
                    for word in blocked_words
                ):
                    continue

                if not self.is_valid_name(
                    text,
                    min_confidence=0.65,
                ):
                    continue

                cleaned = self.clean_name(text)

                if not cleaned or len(cleaned) < 4:
                    continue

            elif value_type == "dob":
                cleaned = self.clean_text(text)

                if not any(
                    re.search(pattern, cleaned)
                    for pattern in self.date_patterns
                ):
                    continue

            else:
                continue

            dy = item["center_y"] - label_y
            abs_dy = abs(dy)

            # PAN fields should be reasonably close to their labels.
            if abs_dy > 250:
                continue

            overlap = self._horizontal_overlap(label_item, item)
            center_distance = abs(
                item["center_x"] - label_item["center_x"]
            )

            # Vertical proximity is useful, but horizontal alignment is
            # especially important when several names exist on the card.
            score = 0.0

            score += max(
                0.0,
                1.0 - (abs_dy / 250.0),
            ) * 35

            score += overlap * 35

            score += float(item["confidence"]) * 20

            # Direction is field-specific on the PAN layout:
            # - Father's Name -> value is typically BELOW the label.
            # - Name -> value is typically ABOVE the OCR-corrupted Name label.
            # - DOB -> value is typically ABOVE the Date of Birth label.
            label_normalized = self.normalize_for_matching(
                label_item["text"]
            )
            is_father_label = (
                "FATHER" in label_normalized
                or "FATHERS" in label_normalized
            )

            if value_type == "name":
                if is_father_label:
                    if dy > 0:
                        score += 25
                    else:
                        score -= 20
                else:
                    if dy < 0:
                        score += 25
                    else:
                        score -= 15
            elif value_type == "dob":
                if dy < 0:
                    score += 15
                else:
                    score -= 10

            # Strongly penalize candidates that are horizontally far away.
            if center_distance > 350:
                score -= 30
            elif center_distance > 200:
                score -= 10

            candidates.append(
                {
                    "score": score,
                    "item": item,
                    "dy": dy,
                    "overlap": overlap,
                }
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda candidate: candidate["score"],
            reverse=True,
        )

        best = candidates[0]

        print(
            f"  -> Spatial candidate for '{label_item['text']}': "
            f"'{best['item']['text']}' "
            f"(dy={best['dy']:.1f}, "
            f"overlap={best['overlap']:.2f}, "
            f"score={best['score']:.1f})"
        )

        return best["item"]

    def extract_names_with_keywords(
        self, text_data: List[Dict]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract applicant and father's names from PAN cards.

        For the tested PAN layout:
          - applicant name is above the Father's Name label
          - father's name is below the Father's Name label
          - OCR may corrupt the applicant "Name" label into text such
            as "EraName"

        The Father's Name label is therefore used as the strongest
        anchor for separating the two names.
        """
        name = None
        father_name = None

        sorted_data = sorted(
            text_data,
            key=lambda item: (item["center_y"], item["center_x"]),
        )

        # ---------------------------------------------------------
        # 1. Locate Father's Name label and extract father's value.
        # ---------------------------------------------------------
        father_label = None

        for item in sorted_data:
            if self.is_father_label(item["text"]):
                father_label = item
                break

        if father_label is not None:
            father_candidate = self._find_value_near_label(
                father_label,
                sorted_data,
                value_type="name",
            )

            if father_candidate:
                father_name = self.clean_name(
                    father_candidate["text"]
                )

                print(
                    f"  -> Label-based father's name: "
                    f"{father_name}"
                )

        # ---------------------------------------------------------
        # 2. Applicant name.
        #
        # Do NOT use a generic "Name" label here. On the tested
        # template OCR recognizes the label as "EraName", and a
        # generic spatial search can therefore select the father's
        # name again.
        #
        # Instead, once Father's Name is known, the applicant name
        # is the strongest valid name immediately ABOVE that label.
        # ---------------------------------------------------------
        if father_label is not None:
            applicant_candidates = []

            for item in sorted_data:
                if item is father_label:
                    continue

                text = item["text"].strip()

                if not text:
                    continue

                if father_name:
                    cleaned = self.clean_name(text)
                    if (
                        cleaned
                        and cleaned.lower() == father_name.lower()
                    ):
                        continue

                normalized = self.normalize_for_matching(text)
                compact = normalized.replace(" ", "")

                blocked_words = [
                    "NAME",
                    "FATHER",
                    "FATHERS",
                    "DATE",
                    "BIRTH",
                    "SIGNATURE",
                    "PERMANENT",
                    "ACCOUNT",
                    "NUMBER",
                    "INCOME",
                    "TAX",
                    "DEPARTMENT",
                    "GOVT",
                    "GOVERNMENT",
                    "INDIA",
                ]

                if any(
                    word in normalized.split() or word in compact
                    for word in blocked_words
                ):
                    continue

                if not self.is_valid_name(
                    text,
                    min_confidence=0.65,
                ):
                    continue

                # Applicant must be above the Father's Name label.
                dy = item["center_y"] - father_label["center_y"]

                if dy >= 0:
                    continue

                # Keep the search local to the field.
                if abs(dy) > 220:
                    continue

                overlap = self._horizontal_overlap(
                    father_label,
                    item,
                )

                # Strong preference for the closest horizontally
                # aligned valid name above Father's Name.
                score = (
                    max(0.0, 1.0 - abs(dy) / 220.0) * 50
                    + overlap * 30
                    + float(item["confidence"]) * 20
                )

                applicant_candidates.append(
                    {
                        "item": item,
                        "score": score,
                    }
                )

            if applicant_candidates:
                applicant_candidates.sort(
                    key=lambda candidate: candidate["score"],
                    reverse=True,
                )

                candidate = applicant_candidates[0]["item"]
                name = self.clean_name(candidate["text"])

                print(
                    f"  -> Applicant name anchored above "
                    f"Father's Name: {name}"
                )

        # ---------------------------------------------------------
        # 3. If the Father's Name label wasn't detected, use the
        # existing Name-label logic as a fallback.
        # ---------------------------------------------------------
        if not name:
            for item in sorted_data:
                if not self.is_name_label(item["text"]):
                    continue

                # Ignore OCR-corrupted labels that are known to
                # represent the Name field itself rather than a
                # useful value anchor.
                normalized = self.normalize_for_matching(
                    item["text"]
                )
                compact = normalized.replace(" ", "")

                if (
                    "ERANAME" in compact
                    or "FATHERS" in compact
                    or "FATHER" in compact
                ):
                    continue

                candidate = self._find_value_near_label(
                    item,
                    sorted_data,
                    value_type="name",
                )

                if candidate:
                    candidate_name = self.clean_name(
                        candidate["text"]
                    )

                    if (
                        father_name
                        and candidate_name.lower()
                        == father_name.lower()
                    ):
                        continue

                    name = candidate_name

                    print(
                        f"  -> Fallback label-based applicant "
                        f"name: {name}"
                    )
                    break

        return name, father_name

    def extract_names_positional(
        self, text_data: List[Dict]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Fallback when OCR does not reliably recognize field labels.

        On a normal PAN card, the applicant name appears above the
        father's name, so valid candidates are sorted vertically.
        """
        valid_candidates = []

        print("\n=== Analyzing candidates for positional extraction ===")

        for item in text_data:
            text = item["text"]

            print(
                f"Checking: '{text}' "
                f"(confidence: {item['confidence']:.2f})"
            )

            # Do not require two words. OCR can return:
            # "DMANIKANDAN" / "DURAISAMY"
            # Reject obvious field labels / OCR fragments before
            # running the generic name heuristic.
            normalized = self.normalize_for_matching(text)
            compact = normalized.replace(" ", "")

            label_words = [
                "NAME",
                "FATHER",
                "FATHERS",
                "DATE",
                "BIRTH",
                "SIGNATURE",
                "PERMANENT",
                "ACCOUNT",
                "NUMBER",
                "INCOME",
                "TAX",
                "DEPARTMENT",
                "GOVT",
                "GOVERNMENT",
                "INDIA",
            ]

            contains_label_word = any(
                word in normalized.split() or word in compact
                for word in label_words
            )

            if contains_label_word:
                print(
                    f"  -> Rejected '{text}' as field-label/OCR noise"
                )
                continue

            if (
                item["confidence"] >= 0.65
                and self.is_valid_name(text, min_confidence=0.65)
            ):
                cleaned_name = self.clean_name(text)

                if cleaned_name and len(cleaned_name) >= 4:
                    valid_candidates.append(
                        {
                            "name": cleaned_name,
                            "center_y": item["center_y"],
                            "center_x": item["center_x"],
                            "confidence": item["confidence"],
                            "item": item,
                        }
                    )
                    print(
                        f"  -> Added to candidates: "
                        f"{cleaned_name}"
                    )

        # Remove duplicates while preserving spatial information.
        unique_candidates = []
        seen = set()

        for candidate in valid_candidates:
            key = candidate["name"].lower()

            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        unique_candidates.sort(
            key=lambda candidate: candidate["center_y"]
        )

        print("\n  -> Final valid name candidates:")

        for candidate in unique_candidates:
            print(
                f"     {candidate['name']} "
                f"(y={candidate['center_y']:.1f}, "
                f"confidence={candidate['confidence']:.2f})"
            )

        name = None
        father_name = None

        if len(unique_candidates) >= 1:
            name = unique_candidates[0]["name"]

        if len(unique_candidates) >= 2:
            father_name = unique_candidates[1]["name"]

        return name, father_name

    def find_names_improved(
        self, text_data: List[Dict]
    ) -> Dict[str, str]:
        """
        Robust PAN name extraction.

        Priority:
        1. Label + spatial relationship
        2. Positional fallback
        3. Sanity checks
        """
        sorted_data = sorted(
            text_data,
            key=lambda item: (item["center_y"], item["center_x"]),
        )

        print("\n=== PAN name extraction ===")

        name, father_name = self.extract_names_with_keywords(
            sorted_data
        )

        if not name or not father_name:
            print(
                "\n=== Label extraction incomplete; "
                "using positional extraction as fallback ==="
            )

            pos_name, pos_father = self.extract_names_positional(
                sorted_data
            )

            if not name and pos_name:
                name = pos_name
                print(f"  -> Assigned positional name: {name}")

            if not father_name and pos_father:
                if not name or pos_father.lower() != name.lower():
                    father_name = pos_father
                    print(
                        f"  -> Assigned positional father's name: "
                        f"{father_name}"
                    )

        if name and father_name and name.lower() == father_name.lower():
            print(
                "  -> Applicant and father's names are identical; "
                "discarding father's name."
            )
            father_name = ""

        return {
            "name": name or "",
            "father_name": father_name or "",
        }

    def find_date_of_birth(
        self, text_data: List[Dict]
    ) -> Optional[str]:
        """
        Find DOB using the Date of Birth label first.
        Falls back to the first valid date if the label is unavailable.
        """
        sorted_data = sorted(
            text_data,
            key=lambda item: (item["center_y"], item["center_x"]),
        )

        # Label-based extraction.
        for item in sorted_data:
            if not self.is_dob_label(item["text"]):
                continue

            candidate = self._find_value_near_label(
                item,
                sorted_data,
                value_type="dob",
            )

            if candidate:
                cleaned = self.clean_text(candidate["text"])

                for pattern in self.date_patterns:
                    match = re.search(pattern, cleaned)

                    if match and self.validate_date(match.group()):
                        print(
                            f"  -> Label-based DOB: {match.group()}"
                        )
                        return match.group()

        # Fallback.
        dates = self.find_dates(sorted_data)

        valid_dates = [
            date for date in dates
            if self.validate_date(date)
        ]

        if valid_dates:
            print(
                f"  -> Fallback DOB: {valid_dates[0]}"
            )
            return valid_dates[0]

        return None

    def validate_pan(self, pan: str) -> bool:
        """Validate the basic structural format of an Indian PAN."""
        if not pan:
            return False

        pan = pan.upper().replace(" ", "")

        return bool(
            re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan)
        )

    def validate_date(self, date_str: str) -> bool:
        # Check if date is valid and reasonable for a birth date
        try:
            for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    current_year = datetime.now().year
                    age = current_year - date_obj.year
                    # Person should be between 1 and 120 years old
                    return 1 <= age <= 120
                except ValueError:
                    continue
            return False
        except:
            return False

    def extract_pan_details(self, image_path: str) -> Dict:
        # Main function that ties everything together
        try:
            # Process the image to make OCR more accurate
            processed_img = self.preprocess_image(image_path)

            # Extract text from the processed image
            text_data = self.extract_text_with_coordinates(processed_img)

            if not text_data:
                return {"error": "No text could be extracted from the image"}

            print("Final extracted text data:")
            for item in text_data:
                print(f"  - {item['text']} (confidence: {item['confidence']:.2f})")

            # Extract different types of information.
            pan_number = self.find_pan_number(text_data)
            names = self.find_names_improved(text_data)
            date_of_birth = self.find_date_of_birth(text_data)

            # Build the final result.
            result = {
                "pan_number": (
                    pan_number
                    if self.validate_pan(pan_number)
                    else None
                ),
                "name": names.get("name", ""),
                "father_name": names.get("father_name", ""),
                "date_of_birth": date_of_birth,
                "extraction_confidence": "low",
                "raw_text": [item["text"] for item in text_data],
            }

            # ---------------------------------------------------------
            # Field-level OCR confidence
            # ---------------------------------------------------------
            field_confidence = {
                "pan_number": 0.0,
                "name": 0.0,
                "father_name": 0.0,
                "date_of_birth": 0.0,
            }

            # PAN confidence.
            if result["pan_number"]:
                normalized_pan = (
                    result["pan_number"]
                    .upper()
                    .replace(" ", "")
                )

                for item in text_data:
                    normalized_text = (
                        self.clean_text(item["text"])
                        .upper()
                        .replace(" ", "")
                    )

                    if normalized_pan == normalized_text:
                        field_confidence["pan_number"] = float(
                            item["confidence"]
                        )
                        break

            # Applicant-name confidence.
            if result["name"]:
                for item in text_data:
                    if (
                        self.clean_name(item["text"]).lower()
                        == result["name"].lower()
                    ):
                        field_confidence["name"] = float(
                            item["confidence"]
                        )
                        break

            # Father's-name confidence.
            if result["father_name"]:
                for item in text_data:
                    if (
                        self.clean_name(item["text"]).lower()
                        == result["father_name"].lower()
                    ):
                        field_confidence["father_name"] = float(
                            item["confidence"]
                        )
                        break

            # DOB confidence.
            if result["date_of_birth"]:
                for item in text_data:
                    if result["date_of_birth"] in self.clean_text(
                        item["text"]
                    ):
                        field_confidence["date_of_birth"] = float(
                            item["confidence"]
                        )
                        break

            result["field_confidence"] = field_confidence

            # ---------------------------------------------------------
            # Completeness score
            # ---------------------------------------------------------
            #
            # IMPORTANT:
            # This is a completeness score, NOT a probability that
            # the extracted values are correct.
            confidence_score = 0

            if result["pan_number"]:
                confidence_score += 40
            if result["name"]:
                confidence_score += 30
            if result["father_name"]:
                confidence_score += 20
            if result["date_of_birth"]:
                confidence_score += 10

            result["confidence_score"] = confidence_score
            result["confidence_score_type"] = "completeness"

            result["extraction_confidence"] = (
                "high"
                if confidence_score >= 90
                else "medium"
                if confidence_score >= 60
                else "low"
            )

            return result

        except Exception as e:
            import traceback

            print(f"Detailed error: {traceback.format_exc()}")
            return {"error": f"Error processing image: {str(e)}"}

    def save_results(
        self, results: Dict, output_path: str = "pan_extraction_results.json"
    ):
        # Save the extraction results to a JSON file for later use
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")


def main():
    # Simple test function to try out the extractor
    extractor = PANCardExtractor()

    image_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/pan_sample/37.jpg"

    try:
        results = extractor.extract_pan_details(image_path)

        print("\n=== PAN Card Extraction Results ===")
        print(f"PAN Number: {results.get('pan_number', 'Not found')}")
        print(f"Name: {results.get('name', 'Not found')}")
        print(f"Father's Name: {results.get('father_name', 'Not found')}")
        print(f"Date of Birth: {results.get('date_of_birth', 'Not found')}")
        print(f"Confidence: {results.get('extraction_confidence', 'Unknown')}")
        print(f"Confidence Score: {results.get('confidence_score', 0)}/100")

        if "error" in results:
            print(f"Error: {results['error']}")

        # Save results to file
        extractor.save_results(results)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()