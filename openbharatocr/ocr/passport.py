import logging
import re
from datetime import datetime
import indiapins

logger = logging.getLogger(__name__)






def preprocess_passport_image(image_path):
    """
    Apply OpenCV CLAHE contrast enhancement and crop lower 30% MRZ region.
    """
    try:
        import cv2
        import numpy as np

        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        elif isinstance(image_path, np.ndarray):
            img = image_path.copy()
        else:
            return None, None

        if img is None:
            return None, None

        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        mrz_crop = enhanced_bgr[int(h * 0.70):, :]

        return enhanced_bgr, mrz_crop
    except Exception:
        return None, None


def ocr_image(image_path):
    """
    Run OCR on a passport image using strictly RapidOCR for maximum speed.
    """
    from openbharatocr.core.engine import _run_rapidocr, _rapidocr_to_text
    
    def extract_fast(img):
        raw = _run_rapidocr(img)
        if raw:
            # lower confidence for MRZ reading to prevent skipping
            return _rapidocr_to_text(raw, confidence_threshold=0.3)
        return ""

    text_main = extract_fast(image_path)
    
    enhanced_img, mrz_crop = preprocess_passport_image(image_path)
    if mrz_crop is not None:
        text_mrz = extract_fast(mrz_crop)
        if text_mrz:
            text_main = (text_main or "") + "\n" + text_mrz

    return text_main, text_main


# ============================================================
# ICAO 9303 CHECK DIGIT & AUTO-CORRECTION HELPERS
# ============================================================

def mrz_char_value(c):
    if '0' <= c <= '9':
        return ord(c) - ord('0')
    if 'A' <= c <= 'Z':
        return ord(c) - ord('A') + 10
    if c == '<':
        return 0
    return 0


def calculate_mrz_check_digit(data_str):
    """
    Calculate ICAO 9303 7-3-1 weighting modulus 10 check digit.
    """
    weights = [7, 3, 1]
    total = sum(mrz_char_value(c) * weights[i % 3] for i, c in enumerate(data_str))
    return str(total % 10)


def correct_mrz_digit_string(s):
    """
    Auto-correct common OCR letter-to-digit errors in numeric MRZ fields.
    """
    trans = str.maketrans({
        'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
        'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'
    })
    return s.translate(trans)


def correct_mrz_letter_string(s):
    """
    Auto-correct common OCR digit-to-letter errors in alpha MRZ fields.
    """
    trans = str.maketrans({
        '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'
    })
    return s.translate(trans)


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_spaces(text):
    return re.sub(r"[ \t]+", " ", text).strip()


def clean_name(text):
    """
    Clean a person's name while preserving spaces.
    """
    text = text.replace("\n", " ")
    text = re.sub(r"[^A-Za-z .'-]", " ", text)
    text = normalize_spaces(text)

    return text.strip(" .-")


def normalize_passport_number(value):
    """
    Normalize a passport number.

    Indian passport numbers are generally one letter followed by
    seven digits. OCR may confuse characters such as O/0, I/1.
    """
    if not value:
        return ""

    value = re.sub(r"[^A-Za-z0-9]", "", value).upper()

    # Typical Indian passport format: C7162010
    match = re.search(r"([A-Z])([0-9]{7})", value)

    if match:
        return match.group(1) + match.group(2)

    return value


# ============================================================
# MRZ
# ============================================================

def find_mrz_lines(text):
    """
    Find the two passport MRZ lines.
    """
    lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip().replace(" ", "")

        if not line:
            continue

        # MRZ lines contain lots of < characters and alphanumeric data.
        if (
            len(line) >= 35
            and (
                line.startswith("P<")
                or line.startswith("P<IND")
                or line.count("<") >= 2
                or re.search(r'[0-9]{6}[0-9][MF<][0-9]{6}', line)
            )
        ):
            lines.append(line)

    # Usually the first matching lines are cleaner (from original image OCR).
    if len(lines) >= 2:
        mrz1, mrz2 = "", ""
        for line in lines:
            if not mrz1 and ("P<" in line or "<<" in line) and not re.search(r'[0-9]{6}[0-9][MF<][0-9]{6}', line):
                mrz1 = line
            elif not mrz2 and re.search(r'[0-9]{6}[0-9][MF<][0-9]{6}', line):
                mrz2 = line
                
            if mrz1 and mrz2:
                break
                
        if mrz1 and mrz2:
            logger.info(f"[MRZ Line Detection] Found line1: '{mrz1}' | line2: '{mrz2}'")
            return mrz1, mrz2
            
        logger.info(f"[MRZ Line Detection] Found fallback line1: '{lines[0]}' | line2: '{lines[1]}'")
        return lines[0], lines[1]

    logger.info("[MRZ Line Detection] No valid MRZ lines detected.")
    return "", ""


def parse_mrz(line1, line2):
    """
    Parse standard TD3 passport MRZ with ICAO 9303 check digit verification
    and auto-correction (O/0, I/1, Z/2, S/5, B/8).
    """

    result = {
        "passport_number": "",
        "surname": "",
        "given_names": "",
        "gender": "",
        "nationality": "",
        "date_of_birth": "",
        "expiry_date": "",
        "date_of_expiry": "",
        "valid": False,
    }

    if not line1 or not line2:
        logger.info("[MRZ Parse Start] Empty MRZ line(s) provided.")
        return result

    logger.info(f"[MRZ Parse Start] Parsing MRZ lines: line1='{line1}', line2='{line2}'")
    line1 = line1.upper().replace(" ", "")
    line2 = line2.upper().replace(" ", "")

    # --------------------------------------------------------
    # Line 1: Surname & Given Names
    # --------------------------------------------------------

    if line1.startswith("P<"):
        name_part = line1[2:]

        # Remove country code if present
        if name_part.startswith("IND"):
            name_part = name_part[3:]
        elif re.match(r'^(?:1ND|I1N|1IN|IN0|1N0|1I1|I11)', name_part):
            name_part = name_part[3:]

        parts = name_part.split("<<", 1)

        surname = parts[0].replace("<", " ").strip()
        given_names = ""
        if len(parts) > 1:
            given_names = parts[1].replace("<", " ").strip()

        result["surname"] = clean_name(surname)
        result["given_names"] = clean_name(given_names)

    # --------------------------------------------------------
    # Line 2: Passport Number (Pos 0..8 + Check Digit at 9)
    # --------------------------------------------------------

    pass_field = line2[:9] if len(line2) >= 9 else ""
    pass_check = line2[9] if len(line2) > 9 else ""

    # Check digit verification and auto-correction
    calc_chk = calculate_mrz_check_digit(pass_field) if pass_field else ""
    if pass_check and calc_chk != pass_check:
        # Try auto-correcting digit portion (chars 1..8)
        corrected_field = pass_field[0] + correct_mrz_digit_string(pass_field[1:])
        if calculate_mrz_check_digit(corrected_field) == pass_check:
            pass_field = corrected_field

    # First char must be a letter; chars 1-7 must be digits
    if pass_field:
        first_char = correct_mrz_letter_string(pass_field[0])
        digits_part = correct_mrz_digit_string(pass_field[1:]).replace("<", "")
        if first_char.isalpha() and len(digits_part) >= 7:
            result["passport_number"] = first_char + digits_part[:7]

    if not result["passport_number"] and len(line2) >= 9:
        result["passport_number"] = normalize_passport_number(line2[:9])

    # --------------------------------------------------------
    # Line 2: Nationality (Pos 10..12)
    # --------------------------------------------------------

    if len(line2) >= 13:
        raw_nat = line2[10:13].replace("<", "")
        nat_corrected = correct_mrz_letter_string(raw_nat)
        if nat_corrected in ['1ND', 'I1N', '1IN', '1N', '11N', 'I1', 'IN0', '1I1', 'I11', 'IND']:
            nat_corrected = 'IND'
        result["nationality"] = nat_corrected
    else:
        result["nationality"] = "IND"

    # --------------------------------------------------------
    # Line 2: Date of Birth (Pos 13..18 + Check Digit at 19)
    # --------------------------------------------------------

    if len(line2) >= 19:
        raw_dob = line2[13:19]
        dob_check = line2[19]
        if calculate_mrz_check_digit(raw_dob) != dob_check:
            raw_dob = correct_mrz_digit_string(raw_dob)
        result["date_of_birth"] = format_mrz_date(raw_dob, is_birth=True)
    elif len(line2) >= 18:
        raw_dob = correct_mrz_digit_string(line2[13:19])
        result["date_of_birth"] = format_mrz_date(raw_dob, is_birth=True)

    # --------------------------------------------------------
    # Line 2: Gender (Pos 20)
    # --------------------------------------------------------

    if len(line2) >= 21:
        gender_char = line2[20]
        if gender_char == "M":
            result["gender"] = "Male"
        elif gender_char == "F":
            result["gender"] = "Female"

    # --------------------------------------------------------
    # Line 2: Date of Expiry (Pos 21..26 + Check Digit at 27)
    # --------------------------------------------------------

    if len(line2) >= 27:
        raw_exp = line2[21:27]
        exp_check = line2[27] if len(line2) > 27 else ""
        if exp_check and calculate_mrz_check_digit(raw_exp) != exp_check:
            raw_exp = correct_mrz_digit_string(raw_exp)
        result["expiry_date"] = format_mrz_date(raw_exp, is_birth=False)
        result["date_of_expiry"] = result["expiry_date"]

    result["valid"] = bool(
        result["passport_number"]
        and result["date_of_birth"]
        and result["expiry_date"]
    )

    logger.info(
        f"[MRZ Parse Output] passport_number='{result['passport_number']}', "
        f"surname='{result['surname']}', given_names='{result['given_names']}', "
        f"dob='{result['date_of_birth']}', gender='{result['gender']}', "
        f"expiry='{result['expiry_date']}', nationality='{result['nationality']}', valid={result['valid']}"
    )

    return result


def format_mrz_date(value, is_birth=False):
    """
    Convert YYMMDD to DD-MM-YYYY.

    For passports, a reasonable century rule is used:
    - birth years in the future are interpreted as 19xx
    - expiry dates are interpreted as 20xx
    """
    try:
        year = int(value[:2])
        month = int(value[2:4])
        day = int(value[4:6])

        if is_birth:
            current_year = datetime.now().year % 100

            if year > current_year:
                full_year = 1900 + year
            else:
                full_year = 2000 + year
        else:
            full_year = 2000 + year

        date_obj = datetime(
            full_year,
            month,
            day,
        )

        return date_obj.strftime("%d-%m-%Y")

    except ValueError:
        return ""


# ============================================================
# VISUAL PASSPORT NUMBER
# ============================================================

def extract_visual_passport_number(text):
    """
    Find passport number from normal OCR.

    Handles examples such as:
        C7162010
        C 7162010
        Passport No. C7162010
    """

    patterns = [
        r"\b[A-Z][0-9]{7}\b",
        r"\b[A-Z]\s*[0-9]{7}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.upper())

        if match:
            return normalize_passport_number(match.group(0))

    return ""


# ============================================================
# VISUAL DATES
# ============================================================

def extract_dates(text):
    """
    Extract DD/MM/YYYY style dates.
    """
    matches = re.findall(
        r"\b(\d{2})[/.-](\d{2})[/.-](\d{4})\b",
        text,
    )

    dates = []

    for day, month, year in matches:
        try:
            date_obj = datetime(
                int(year),
                int(month),
                int(day),
            )

            dates.append(
                date_obj.strftime("%d-%m-%Y")
            )

        except ValueError:
            continue

    # Preserve order but remove duplicates
    result = []

    for date in dates:
        if date not in result:
            result.append(date)

    return result


# ============================================================
# FRONT PASSPORT FIELDS
# ============================================================

def extract_front_fields(text, passport_number=""):
    """
    Extract fields visible on the front page.
    """
    logger.info("[Visual Front Fields Start] Extracting front visual fields from OCR text")

    result = {
        "surname": "",
        "given_names": "",
        "gender": "",
        "nationality": "",
        "date_of_birth": "",
        "expiry_date": "",
        "date_of_expiry": "",
        "place_of_birth": "",
        "place_of_issue": "",
        "date_of_issue": "",
    }

    lines = [
        normalize_spaces(line)
        for line in text.splitlines()
        if line.strip()
    ]

    def is_static_label(line):
        lower = line.lower()
        static_terms = [
            "republic of india", "passport", "type", "country code",
            "surname", "given name", "nationality", "sex", "date of birth",
            "place of birth", "place of issue", "date of issue", "date of expiry",
            "placeot birth", "place oue", "dateofiss", "datedfexpiry"
        ]
        return any(term in lower for term in static_terms)

    for i, line in enumerate(lines):
        lower = line.lower()

        # ----------------------------------------------------
        # Place of Birth
        # ----------------------------------------------------
        pob_match = re.search(
            r"(?:place\s*o[ft]\s*birth|place\s*ot?\s*birth|placeot\s*birth"
            r"|place\s*birth|place.*birth|place.*bicth"
            r"|-er/placeot\s*birth|-er/placeot|place\s*o[ft]\s*bir|place\s*ot?\s*bir"
            r"|placeof\s*birth|placeofbirth|\/place[^\s]*birth)",
            line,
            re.I
        )
        if pob_match and not result["place_of_birth"]:
            after = line[pob_match.end():].strip(" :-./")
            if after:
                cand = clean_place(after)
                if cand and len(cand) >= 2:
                    result["place_of_birth"] = cand
            if not result["place_of_birth"]:
                for offset in (1, 2):
                    if i + offset < len(lines):
                        cand_line = lines[i + offset]
                        if cand_line and not is_static_label(cand_line) and not is_label_line(cand_line):
                            cand = clean_place(cand_line)
                            if cand and len(cand) >= 2 and not re.search(r'\d{2}[/.-]\d{2}', cand):
                                result["place_of_birth"] = cand
                                break

        # ----------------------------------------------------
        # Place of Issue
        # ----------------------------------------------------
        poi_match = re.search(
            r"(?:place\s*o[ft]\s*issue|place\s*ot?\s*issue|place\s*of\s*theue"
            r"|place\s*issue|gve/place\s*oue|gve/place|place\s*oue"
            r"|place\s*o[ft]\s*is|place\s*ot?\s*is"
            r"|placeof\s*issue|placeofissue|placess?ue|place[aeiou]*?ss?ue"
            r"|place.{0,4}fl.{0,4}ss?ue|place.{0,4}[io]ss?ue"
            r"|\/place[^\s]*iss?ue|\/place[^\s]*ss?ue"
            r"|fsue|cefsue)",
            line,
            re.I
        )
        if poi_match and not result["place_of_issue"]:
            after = line[poi_match.end():].strip(" :-./")
            if after:
                cand = clean_place(after)
                if cand and len(cand) >= 2:
                    result["place_of_issue"] = cand
            if not result["place_of_issue"]:
                for offset in (1, 2):
                    if i + offset < len(lines):
                        cand_line = lines[i + offset]
                        if cand_line and not is_static_label(cand_line) and not is_label_line(cand_line):
                            cand = clean_place(cand_line)
                            if cand and len(cand) >= 2 and not re.search(r'\d{2}[/.-]\d{2}', cand):
                                result["place_of_issue"] = cand
                                break

        # ----------------------------------------------------
        # Surname & Given Names
        # ----------------------------------------------------
        if "surname" in lower or "sumame" in lower:
            if i + 1 < len(lines):
                cand = lines[i + 1]
                if not is_label_line(cand) and not is_static_label(cand):
                    result["surname"] = clean_name(cand)
        elif ("given name" in lower or "given maen" in lower or "given" in lower) and not result["given_names"]:
            if i + 1 < len(lines):
                cand = lines[i + 1]
                if not is_label_line(cand) and not is_static_label(cand):
                    result["given_names"] = clean_name(cand)

        # ----------------------------------------------------
        # Nationality & Gender
        # ----------------------------------------------------
        if "nationality" in lower or "indian" in lower:
            if "indian" in lower or "ind" in lower:
                result["nationality"] = "IND"
            
        if "sex" in lower or "gender" in lower or "fait/sex" in lower:
            if re.search(r'\b[f]\b|female', lower):
                result["gender"] = "Female"
            elif re.search(r'\b[m]\b|male', lower) or lower.endswith("m"):
                result["gender"] = "Male"

    # --------------------------------------------------------
    # Dates extraction with chronological sorting
    # --------------------------------------------------------
    all_dates = extract_dates(text)

    # Date of Birth
    dob_match = re.search(r'(?:date of birth|dob|birth|bith)[^\n]{0,30}?(\d{2}[/.-]\d{2}[/.-]\d{4})', text, re.I)
    if not dob_match:
        dob_match = re.search(r'(?:date of birth|dob|birth|bith)(?:.*?\n){1,2}.*?(\d{2}[/.-]\d{2}[/.-]\d{4})', text, re.I)
    
    if dob_match:
        result["date_of_birth"] = normalize_date(dob_match.group(1))


    # Explicit labels
    pob_patterns = [
        r"(?:place\s*of\s*birth|placeofbirth|placeof\s*birth|irth|ofbirth)[^\n]{0,20}?\n\s*([A-Za-z\s,.-]+)",
    ]
    result["place_of_birth"] = extract_labeled_person(text, pob_patterns)

    poi_patterns = [
        r"(?:place\s*of\s*issue|placeofissue|placeof\s*issue|ssue|oissuo|oissue|fsue|cefsue)[^\n]{0,20}?\n\s*([A-Za-z\s,.-]+)",
    ]
    result["place_of_issue"] = extract_labeled_person(text, poi_patterns)

    # Explicit Date of Issue regex match
    issue_match = re.search(r'(?:date of issue|date of is|dateofiss)[^\n]{0,30}?(\d{2}[/.-]\d{2}[/.-]\d{4})', text, re.I)
    if issue_match:
        result["date_of_issue"] = normalize_date(issue_match.group(1))

    # Explicit Expiry Date regex match
    exp_match = re.search(r'(?:expiry|date of exp|datedfexpiry)[^\n]{0,30}?(\d{2}[/.-]\d{2}[/.-]\d{4})', text, re.I)
    if exp_match:
        result["expiry_date"] = normalize_date(exp_match.group(1))

    # Chronological sort for issue vs expiry from all_dates
    non_dob_dates = [d for d in all_dates if d != result["date_of_birth"]]
    if len(non_dob_dates) >= 2:
        parsed_dates = []
        for d_str in non_dob_dates:
            try:
                dt = datetime.strptime(d_str, "%d-%m-%Y")
                parsed_dates.append((dt, d_str))
            except ValueError:
                continue
        parsed_dates.sort(key=lambda x: x[0])
        if len(parsed_dates) >= 2:
            if not result["date_of_issue"]:
                result["date_of_issue"] = parsed_dates[0][1]
            if not result["expiry_date"]:
                result["expiry_date"] = parsed_dates[-1][1]

    result["date_of_expiry"] = result["expiry_date"]

    # Global regex search for gender if not found per-line
    if not result["gender"]:
        if re.search(r'\b(?:sex|gender)\s*[:;-]?\s*m\b', text, re.I) or re.search(r'\bmale\b', text, re.I) or re.search(r'\bM\s+\d{2}/\d{2}/\d{4}\b', text):
            result["gender"] = "Male"
        elif re.search(r'\b(?:sex|gender)\s*[:;-]?\s*f\b', text, re.I) or re.search(r'\bfemale\b', text, re.I) or re.search(r'\bF\s+\d{2}/\d{2}/\d{4}\b', text):
            result["gender"] = "Female"
            
    if not result["nationality"]:
        if re.search(r'\bindian\b|\bIND\b', text, re.I):
            result["nationality"] = "IND"

    # Positional fallback for place_of_birth and place_of_issue
    if not result["place_of_birth"] or not result["place_of_issue"]:
        dob_idx = -1
        date_line_idx = len(lines)
        for idx, line in enumerate(lines):
            if result["date_of_birth"] and (result["date_of_birth"] in line or result["date_of_birth"].replace("-", "/") in line):
                dob_idx = idx
            elif "P<" in line or "<<" in line:
                if idx < date_line_idx:
                    date_line_idx = idx

        place_candidates = []
        start_search = dob_idx + 1 if dob_idx != -1 else 0
        noise_place_words = [
            "republic", "india", "code", "type", "passpon", "passport", "chq",
            "gst", "mode", "tgs", "neft", "expiry", "date", "pay", "given", "name", "surname", "nationality", "sex", "country"
        ]
        for line in lines[start_search:date_line_idx]:
            lower_ln = line.lower()
            if is_static_label(line) or is_label_line(line) or "<" in line:
                continue
            if any(nw in lower_ln for nw in noise_place_words):
                continue
            if result.get("passport_number") and result["passport_number"] in line:
                continue
            if re.search(r'\b(P\s*IND|IND)\b', line) or line.startswith('/'):
                continue
            if re.search(r'\d{2}[/.-]\d{2}[/.-]\d{4}', line):
                continue
            cleaned = clean_place(line)
            if cleaned and len(cleaned) >= 3 and not re.search(r'^[0-9]+$', cleaned):
                place_candidates.append(cleaned)

        if place_candidates:
            if not result["place_of_birth"] and len(place_candidates) >= 1:
                result["place_of_birth"] = place_candidates[0]
            if not result["place_of_issue"] and len(place_candidates) >= 2:
                result["place_of_issue"] = place_candidates[1]

    # Smart Visual Name Extraction (Fallback if labels missed)
    if (
        not result["surname"] 
        or not result["given_names"] 
        or "KKK" in result["surname"]
        or "EEE" in result["surname"]
        or "KKK" in result["given_names"]
        or "EEE" in result["given_names"]
        or len(result["surname"]) > 20
        or len(result["given_names"]) > 30
    ) and passport_number and result["date_of_birth"]:
        p_idx = -1
        d_idx = -1
        for i, line in enumerate(lines):
            if passport_number in line and "<" not in line:
                p_idx = i
            dob_parts = result["date_of_birth"].split("-")
            if len(dob_parts) == 3:
                dob_slash = f"{dob_parts[0]}/{dob_parts[1]}/{dob_parts[2]}"
                if dob_slash in line or result["date_of_birth"] in line:
                    d_idx = i
        
        if d_idx == -1 and len(dob_parts) == 3:
            for i, line in enumerate(lines):
                if i > p_idx and dob_parts[2] in line:
                    d_idx = i
                    break

        if 0 <= p_idx < d_idx:
            name_candidates = []
            for line in lines[p_idx + 1:d_idx]:
                words = re.findall(r'\b[A-Z]{2,}\b', line)
                lower_words = re.findall(r'\b[a-z]{3,}\b', line)
                
                if len(words) > 0 and len(lower_words) <= 2:
                    ignore_words = {"INDIAN", "REPUBLIC", "INDIA", "CODE", "TYPE", "SURNAME", "NAME", "GIVEN", "PASSPORT", "SEX", "DATE", "BIRTH", "PLACE", "ISSUE", "EXPIRY", "OF", "MAENETS", "MAEN"}
                    clean_cands = [w for w in words if w not in ignore_words]
                    if clean_cands:
                        name_candidates.append(" ".join(clean_cands))
                        
            if name_candidates:
                result["surname"] = name_candidates[0]
                if len(name_candidates) > 1:
                    result["given_names"] = " ".join(name_candidates[1:])

    logger.info(f"[Visual Front Fields Output] date_of_issue='{result['date_of_issue']}', place_of_issue='{result['place_of_issue']}', gender='{result['gender']}', surname='{result['surname']}', given_names='{result['given_names']}', dob='{result['date_of_birth']}', expiry='{result['expiry_date']}', nationality='{result['nationality']}'")

    return result


# ============================================================
# BACK PASSPORT FIELDS
# ============================================================

def is_address_stop_line(line):
    lower = line.lower()
    clean_line = line.replace(" ", "").upper()

    stop_keywords = [
        "old passport", "passpor o.", "passpor", "file no", "fileno", "file.no",
        "file_no", "r./file", "file number", "a/ld passpor", "old pass"
    ]
    if any(kw in lower for kw in stop_keywords):
        return True

    # File number / Old passport detection (e.g. BN79C5034910118, BNM067214376124, G4343573)
    if re.search(r'\b[A-Z]{1,4}[0-9]{7,14}\b', clean_line):
        if not re.search(r'\b\d{6}\b', line) and not any(w in lower for w in ["nagar", "street", "road", "cross", "stage", "block", "city", "karnataka"]):
            return True

    if "rs." in lower or "one lakh" in lower or "thousand" in lower:
        return True

    return False


def is_back_label_line(line):
    lower = line.lower()
    labels = [
        "father", "mother", "spouse", "guardian", "legal",
        "address", "file", "old passport", "passport no", "emigration", "check", "required"
    ]
    return any(label in lower for label in labels)


def extract_back_fields(text):
    """
    Extract Indian passport back-page information.
    """
    # Fix common OCR mistakes in PIN code (like 'S' instead of '5', 'O' instead of '0')
    def fix_pin(match):
        prefix = match.group(1)
        pin_str = match.group(2).replace(' ', '')
        subs = {'S': '5', 's': '5', 'O': '0', 'o': '0', 'B': '8', 'b': '8', 'l': '1', 'I': '1', 'Z': '2', 'z': '2'}
        for k, v in subs.items():
            pin_str = pin_str.replace(k, v)
        return prefix + pin_str
    
    text = re.sub(r'\b(PIN\s*(?:CODE)?\s*[:.-]?\s*)([SBOlIZsboliz\d\s]{6,9})\b', fix_pin, text, flags=re.I)

    result = {
        "passport_number": "",
        "father_name": "",
        "mother_name": "",
        "spouse_name": "",
        "address": {
            "raw": "",
            "lines": [],
            "pincode": "",
        },
        "city": "",
        "file_number": "",
    }

    lines = [
        normalize_spaces(line)
        for line in text.splitlines()
        if line.strip()
    ]

    result["passport_number"] = extract_visual_passport_number(text)

    # --------------------------------------------------------
    # Father / Legal Guardian
    # --------------------------------------------------------
    father_header_re = re.compile(
        r"(?:name\s*o[ft]\s*father|father\s*name|name\s*offather|legal\s*guardian|raran|guardian)",
        re.I
    )

    for i, line in enumerate(lines):
        if father_header_re.search(line) and not result["father_name"]:
            m = father_header_re.search(line)
            after = line[m.end():].strip(" :-./")
            cand = clean_name(after)
            if cand and looks_like_person_name(cand):
                result["father_name"] = cand.title()
                break
            for offset in (1, 2, 3):
                if i + offset < len(lines):
                    cand_line = lines[i + offset]
                    if cand_line and not is_back_label_line(cand_line):
                        cand = clean_name(cand_line)
                        if cand and looks_like_person_name(cand) and cand.upper() != result["passport_number"]:
                            result["father_name"] = cand.title()
                            break
            if result["father_name"]:
                break

    # --------------------------------------------------------
    # Mother
    # --------------------------------------------------------
    mother_header_re = re.compile(
        r"(?:name\s*o[ft]\s*mother|mother\s*name|nameof\s*mother|fothe)",
        re.I
    )

    for i, line in enumerate(lines):
        if mother_header_re.search(line) and not result["mother_name"]:
            m = mother_header_re.search(line)
            after = line[m.end():].strip(" :-./")
            cand = clean_name(after)
            if cand and looks_like_person_name(cand):
                result["mother_name"] = cand.title()
                break
            for offset in (1, 2):
                if i + offset < len(lines):
                    cand_line = lines[i + offset]
                    if cand_line and not is_back_label_line(cand_line):
                        cand = clean_name(cand_line)
                        if cand and looks_like_person_name(cand) and cand.upper() != result["passport_number"]:
                            result["mother_name"] = cand.title()
                            break
            if result["mother_name"]:
                break

    # --------------------------------------------------------
    # Spouse
    # --------------------------------------------------------
    spouse_header_re = re.compile(
        r"(?:name\s*o[ft]\s*spouse|spouse\s*name)",
        re.I
    )

    for i, line in enumerate(lines):
        if spouse_header_re.search(line) and not result["spouse_name"]:
            m = spouse_header_re.search(line)
            after = line[m.end():].strip(" :-./")
            cand = clean_name(after)
            if cand and looks_like_person_name(cand):
                result["spouse_name"] = cand.title()
                break
            for offset in (1, 2):
                if i + offset < len(lines):
                    cand_line = lines[i + offset]
                    if cand_line and not is_back_label_line(cand_line):
                        cand = clean_name(cand_line)
                        if cand and looks_like_person_name(cand) and cand.upper() != result["passport_number"]:
                            result["spouse_name"] = cand.title()
                            break
            if result["spouse_name"]:
                break

    # --------------------------------------------------------
    # Parent/Spouse Fallback if headers garbled (scanning lines before Address)
    # --------------------------------------------------------
    addr_idx = len(lines)
    for i, line in enumerate(lines):
        if "address" in line.lower() or "pin" in line.lower():
            addr_idx = i
            break

    if not result["father_name"] or not result["mother_name"]:
        name_candidates = []
        for line in lines[:addr_idx]:
            if is_back_label_line(line):
                continue
            cand = clean_name(line)
            if cand and looks_like_person_name(cand) and cand.upper() != result["passport_number"]:
                if not any(w in cand.upper() for w in ["EMIGRATION", "CHECK", "REQUIRED", "BANK", "ICICI", "LTD", "GST"]):
                    name_candidates.append(cand.title())

        if name_candidates:
            if not result["father_name"] and len(name_candidates) >= 1:
                result["father_name"] = name_candidates[0]
            if not result["mother_name"] and len(name_candidates) >= 2:
                result["mother_name"] = name_candidates[1]
            if not result["spouse_name"] and len(name_candidates) >= 3:
                result["spouse_name"] = name_candidates[2]

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------
    address_start = None

    for i, line in enumerate(lines):
        lower = line.lower()
        if "address" in lower and "old passport" not in lower:
            address_start = i
            break

    if address_start is None:
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if re.search(r'\b(?:pin|pincode|district|bhavan|marg)\b', line, re.I) or re.search(r'\b\d{6}\b', line):
                address_start = max(0, i - 4)
                break

    if address_start is not None:
        address_lines = []

        for line in lines[address_start + 1:]:
            if is_address_stop_line(line):
                if len(address_lines) == 0:
                    continue
                break
    
    address_idx = address_start if address_start is not None else -1
    address_started = (address_idx != -1)
    address_lines = []

    # --------------------------------------------------------
    # PIN
    # --------------------------------------------------------
    pin_match = re.search(r"\bPIN\s*[:.-]?\s*(\d{6})\b", text, re.I)
    if not pin_match:
        pin_match = re.search(r"\b(\d{6})\b", text)

    pincode = ""
    if pin_match:
        pincode = pin_match.group(1)
        result["address"]["pincode"] = pincode

    if address_started:
        for line in lines[address_idx:]:
            if is_address_stop_line(line) and len(address_lines) > 0:
                break
            
            # Additional heuristic: If we hit a line that looks like a completely different section
            # e.g. a date or file number appearing AFTER we've collected address lines
            if len(address_lines) > 0:
                if re.match(r'^\d{2}[/.-]\d{2}[/.-]\d{4}$', line.strip()):
                    break
                if re.match(r'^[A-Z]{2,4}[0-9]{8,15}$', line.strip()):
                    break

            if is_address_line(line):
                cleaned_line = clean_address_line(line)
                if cleaned_line:
                    address_lines.append(cleaned_line)

            if len(address_lines) >= 8:
                break

        if address_lines:
            # Fix inverted OCR lines (upside down image) by enforcing logical address ordering
            def addr_sort_key(line):
                upper = line.upper()
                if re.search(r'\b(?:PIN|INDIA|KARNATAKA|STATE)\b', upper) or re.search(r'\b\d{6}\b', line):
                    return 2  # Bottom
                if re.match(r'^(?:NO\.?|#|PLOT|FLAT|ROOM|DOOR|HOUSE)\b', upper):
                    return 0  # Top
                return 1      # Middle
            
            address_lines.sort(key=addr_sort_key)
            result["address"]["lines"] = address_lines
            raw_address = "\n".join(address_lines)
            raw_address = re.sub(r'\s+[A-Z]$', '', raw_address, flags=re.IGNORECASE)
            result["address"]["raw"] = raw_address

            city = ""
            if pincode:
                try:
                    pin_data = indiapins.matching(pincode)
                    if pin_data and len(pin_data) > 0:
                        city = pin_data[0].get('District', '')
                except Exception:
                    pass

            if not city:
                for kw in ['DISTRICT', 'CITY', 'TOWN']:
                    for line in address_lines:
                        m = re.search(r'([A-Za-z\s]+?)\s*' + kw + r'\b', line, re.I)
                        if m:
                            parts = m.group(1).split(',')
                            city = parts[-1].strip()
                            break
                    if city:
                        break

            if not city:
                m2 = re.search(r'([A-Za-z \t]+)(?:,\s*)?\bPIN', raw_address, re.I)
                if m2:
                    parts = m2.group(1).split(',')
                    city = parts[-1].strip().split()[-1]
                else:
                    for i, line in enumerate(address_lines):
                        if re.search(r'\b\d{6}\b', line) or 'PIN' in line.upper():
                            if i > 0 and not re.search(r'\b(?:NO|#|PLOT|FLAT|ROOM|DOOR|HOUSE)\b', address_lines[i-1], re.I) and not bool(re.search(r'\d+', address_lines[i-1])):
                                parts = address_lines[i-1].split(',')
                                city = parts[-1].strip()
                            elif i + 1 < len(address_lines):
                                parts = address_lines[i+1].split(',')
                                city = parts[-1].strip()
                            elif i > 0:
                                parts = address_lines[i-1].split(',')
                                city = parts[-1].strip()
                            break
            
            if city:
                city = re.sub(r'\b\d{6}\b', '', city).strip()
                city = re.sub(r'\s+[a-z][a-z.\s]*$', '', city).strip()
                city_words = city.split()
                city_words = [w for w in city_words if not (re.sub(r'[^a-zA-Z]', '', w).islower() and len(re.sub(r'[^a-zA-Z]', '', w)) <= 3)]
                city = ' '.join(city_words)
            result["city"] = city

    # --------------------------------------------------------
    # File number
    # --------------------------------------------------------
    file_patterns = [
        r"(?:file\s*(?:no|number|num)?\.?\s*|\br\./file\s*no\.?\s*)([A-Z0-9]{9,16})",
        r"\b([A-Z]{2,4}[0-9]{7,14})\b",
    ]

    for pattern in file_patterns:
        matches = re.finditer(pattern, text, re.I)
        for match in matches:
            value = match.group(1).upper()
            if value != result["passport_number"] and len(value) >= 9 and re.search(r'\d{5,}', value):
                if not re.fullmatch(r'\d{6}', value) and not re.search(r'^\d{2}/\d{2}/\d{4}$', value):
                    result["file_number"] = value
                    break
        if result["file_number"]:
            break



    return result


def extract_labeled_person(text, patterns):
    """
    Extract a person's name after a known passport label.
    """

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.I | re.S,
        )

        if match:
            value = clean_name(match.group(1))

            value = re.sub(
                r"\s+",
                " ",
                value,
            ).strip()

            if looks_like_person_name(value):
                return value.title()

    return ""


# ============================================================
# CLEANING
# ============================================================

def clean_place(value):
    value = re.sub(
        r"[^A-Za-z0-9, .'-]",
        " ",
        value,
    )

    value = normalize_spaces(value)
    
    # Filter out OCR noise (lowercase artifacts, or words with abnormally low vowel density typically caused by Tesseract hallucinating regional scripts)
    words = value.split()
    clean_words = []
    for w in words:
        # Skip standalone lowercase noise
        clean_w = re.sub(r'[^a-zA-Z]', '', w)
        if clean_w and clean_w.islower() and len(clean_w) <= 2:
            continue
            
        # Check vowel density for long words to filter out garbled regional script
        alpha_w = re.sub(r'[^A-Za-z]', '', w)
        if len(alpha_w) > 7:
            vowels = len(re.findall(r'[AEIOUYaeiouy]', alpha_w))
            # If less than 25% vowels in a long word, it's likely gibberish (e.g. PRBPRRGRIESIL)
            if vowels / len(alpha_w) < 0.25:
                continue
                
        clean_words.append(w)

    value = " ".join(clean_words)
    return value.strip(" .,-")


def clean_address_line(value):
    # Handle pipe-delimited chunks
    if "|" in value:
        chunks = [c.strip() for c in value.split("|")]
        valid_chunks = [c for c in chunks if c and is_address_line(c)]
        value = " ".join(valid_chunks)
        if not value.strip():
            return ""

    value = re.sub(
        r"[^A-Za-z0-9, ./#'()-]",
        " ",
        value,
    )
    
    value = normalize_spaces(value).strip()

    # Remove file-number tokens (e.g. BNMO067214376124)
    value = re.sub(r'\b[A-Z]{2,4}[0-9]{8,}\b', '', value)
    
    # Fix RapidOCR missing spaces
    # Insert space between letters and numbers
    value = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', value)
    value = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', value)
    
    # Heuristic fixes for common OCR misreads in addresses
    value = re.sub(r'\bN\s*0(?=\d)', 'NO. ', value, flags=re.IGNORECASE)
    value = re.sub(r'\b0\s*AD\b', 'ROAD', value, flags=re.IGNORECASE)
    value = re.sub(r'\bBMR\s*ROAD\b', 'BM ROAD', value, flags=re.IGNORECASE)
    value = re.sub(r'^.*?/?[A-Za-z]+ PIN\b', 'PIN', value, flags=re.IGNORECASE)
    
    # Insert space before common address keywords if concatenated
    for kw in ['ROAD', 'CROSS', 'NAGAR', 'MOHALLA', 'LAYOUT', 'STREET', 'CITY', 'DISTRICT', 'TALUK', 'MAIN', 'PHASE', 'STAGE']:
        # Ensure it doesn't split if already spaced, and only splits if preceded/followed by a letter
        value = re.sub(rf'([a-zA-Z])({kw})', rf'\1 \2', value, flags=re.IGNORECASE)
        value = re.sub(rf'({kw})([a-zA-Z])', rf'\1 \2', value, flags=re.IGNORECASE)
    
    # Filter out lowercase noise at the end of lines
    words = value.split()
    clean_words = []
    for w in words:
        clean_w = re.sub(r'[^a-zA-Z]', '', w)
        if clean_w and clean_w.islower() and len(clean_w) <= 3:
            continue
        clean_words.append(w)
        
    value = " ".join(clean_words)
    return value.strip(" .-")


def is_address_line(line):
    # Handle pipe-delimited chunks
    if "|" in line:
        chunks = [c.strip() for c in line.split("|")]
        return any(is_address_line(c) for c in chunks if c)

    lower = line.lower()

    if not line:
        return False

    if "old passport" in lower:
        return False

    if "file no" in lower:
        return False

    if "passport no" in lower:
        return False
        
    if re.match(r"^[A-Z]{3,4}[0-9]{8,12}$", line.replace(" ", "").upper()):
        return False

    # Address lines usually contain numbers, commas,
    # locations, PIN, road/cross etc.
    address_keywords = [
        "cross",
        "road",
        "street",
        "layout",
        "mohalla",
        "nagar",
        "city",
        "district",
        "karnataka",
        "india",
        "pin",
        "no.",
        "no ",
        "town",
        "village",
        "post",
        "taluk",
        "bhavan",
        "marg"
    ]

    # Reject dates
    if re.search(r'^\d{2}/\d{2}/\d{4}$', line.strip()):
        return False
        
    # Reject common labels
    if any(label in lower for label in ["father", "mother", "spouse", "emigration", "check", "required", "signature"]):
        return False

    has_keyword = any(keyword in lower for keyword in address_keywords)
    has_number = bool(re.search(r"\d", line))
    has_comma = "," in line
    
    if has_keyword:
        return True
    if has_number:
        return True
    if has_comma and len(re.sub(r'[^a-zA-Z]', '', line)) > 5:
        return True
        
    return False


def is_label_line(line):
    lower = line.lower()

    labels = [
        "surname",
        "given name",
        "date of birth",
        "place of birth",
        "place of issue",
        "date of issue",
        "date of expiry",
        "passport",
        "republic of india",
    ]

    return any(label in lower for label in labels)


def looks_like_person_name(value):
    if not value:
        return False

    words = value.split()

    if not words:
        return False

    # Reject obvious OCR labels
    bad_words = [
        "passport",
        "address",
        "name",
        "father",
        "mother",
        "spouse",
        "guardian",
        "legal",
        "date",
        "place",
        "issue",
        "expiry",
        "file",
        "bank",
        "icici",
        "ltd",
        "gst",
        "rs",
        "chq",
        "seller",
        "signature",
        "mode",
        "neft",
        "tgs",
        "pay",
        "road",
    ]

    lowered = value.lower()

    if any(word in lowered for word in bad_words):
        return False

    # Names should contain letters
    if not re.search(r"[A-Za-z]", value):
        return False

    return True


def normalize_date(value):
    try:
        value = value.replace(".", "/").replace("-", "/")

        date_obj = datetime.strptime(
            value,
            "%d/%m/%Y",
        )

        return date_obj.strftime("%d-%m-%Y")

    except ValueError:
        return ""


# ============================================================
# MAIN EXTRACTION
# ============================================================

def find_best_passport_number(front_text, back_text, mrz_data):
    """
    Find the best passport number matching [A-Z][0-9]{7}.
    Considers both front and back text, MRZ, labels, and corrects OCR corruption.
    """
    import collections
    
    candidates = collections.defaultdict(lambda: {"score": 0, "contexts": set()})
    
    def add_candidate(cand, score_boost, context_name):
        if not cand: return
        # ensure it's 8 chars
        if len(cand) == 8 and re.match(r'^[A-Z]{1,2}[0-9]{6,7}$', cand):
            candidates[cand]["score"] += score_boost
            candidates[cand]["contexts"].add(context_name)
            
    # 1. Look at MRZ data
    if mrz_data and mrz_data.get("passport_number"):
        val = mrz_data["passport_number"]
        if re.match(r"^[A-Z]{1,2}[0-9]{6,7}$", val) and len(val) == 8:
            add_candidate(val, 50, "mrz_parsed")
            
    # 2. Search all texts for patterns
    ocr_corrections = {
        '€': 'C', '©': 'C', '(': 'C', '[': 'C', '{': 'C', '<': 'C', '¢': 'C',
        '$': 'S', '5': 'S',
        '@': 'A',
        '8': 'B',
        '0': 'O',
        '1': 'I', '|': 'I', '!': 'I',
    }

    pattern = r'(?<!\d)([^\s\d])\s*([0-9]{7})(?!\d)'
    
    def process_text(text, source_name):
        for match in re.finditer(pattern, text.upper()):
            first_char = match.group(1)
            digits = match.group(2)
            
            if first_char in ocr_corrections:
                first_char = ocr_corrections[first_char]
                
            cand = first_char + digits
            
            start_idx = match.start()
            window = text[max(0, start_idx - 40):start_idx].upper()
            
            score = 10
            contexts = [source_name]
            
            if re.search(r'PASSPORT\s*(NO|NUMBER)', window):
                score += 50
                contexts.append("label_passport_no")
            elif 'P IND' in window or 'P<IND' in window:
                score += 30
                contexts.append("label_p_ind")
            
            line = text[max(0, text.rfind('\n', 0, start_idx)):text.find('\n', start_idx) if text.find('\n', start_idx) != -1 else len(text)].upper()
            if '<' in line and len(line) > 30:
                score += 40
                contexts.append("mrz_line")
                
            add_candidate(cand, score, source_name)
            for ctx in contexts:
                candidates[cand]["contexts"].add(ctx)

    process_text(front_text, "front")
    if back_text:
        process_text(back_text, "back")

    # 2b. Label-based search: find "Passport No" label and grab the next line.
    # This handles RapidOCR output where the number appears on its own line after the label.
    def label_based_search(text, source_name):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            upper = line.upper().replace(" ", "").replace(".", "")
            if "PASSPORTNO" in upper or "PASSPORTNO." in upper or "PASSPORNO" in upper:
                # Check next 1-4 lines for a passport-number-shaped token
                for offset in range(1, 5):
                    if i + offset >= len(lines):
                        break
                    candidate_line = lines[i + offset]
                    # Try direct match [A-Z]{1,2}[0-9]{6,7}
                    m = re.search(r'\b([A-Z]{1,2}[0-9]{6,7})\b', candidate_line.upper())
                    if m and len(m.group(1)) == 8:
                        add_candidate(m.group(1), 60, f"{source_name}_label")
                        break
                    
                    # Try treating as OCR merging adjacent char — take chars 1-7 if it has 8 chars
                    # But if the valid number has 8 chars anyway, it's matched above.
                    # We can fallback to extracting the last 7 chars if the first char was noise and it had 9 chars.
                    m2 = re.search(r'\b([A-Z]{2,3}[0-9]{6,7})\b', candidate_line.upper())
                    if m2:
                        token = m2.group(1)
                        # Example: A U 286932
                        if len(token) > 8:
                            # Trim leading character
                            token_trimmed = token[-8:]
                            if re.match(r'^[A-Z]{1,2}[0-9]{6,7}$', token_trimmed):
                                add_candidate(token_trimmed, 30, f"{source_name}_label_corrected")
                        break

    label_based_search(front_text, "front")
    if back_text:
        label_based_search(back_text, "back")
    
    # 3. Handle non-alpha candidates
    merged = collections.defaultdict(lambda: {"score": 0, "contexts": set()})
    
    for cand, data in candidates.items():
        if cand[0].isalpha():
            merged[cand]["score"] += data["score"]
            merged[cand]["contexts"].update(data["contexts"])
        else:
            # For non-alpha, find best matching alpha candidate
            digits = re.sub(r'^[A-Z]+', '', cand)
            matched_alpha = None
            for other_cand in candidates:
                if other_cand[0].isalpha() and other_cand.endswith(digits):
                    matched_alpha = other_cand
                    break
            
            if matched_alpha:
                merged[matched_alpha]["score"] += data["score"]
                merged[matched_alpha]["contexts"].update(data["contexts"])
            else:
                if data["score"] >= 40:
                    # Coerce first char to C
                    coerced = 'C' + cand[1:]
                    merged[coerced]["score"] += data["score"]
                    merged[coerced]["contexts"].update(data["contexts"])

    # 4. Find the best candidate
    best_cand = None
    best_score = -1
    
    for cand, data in merged.items():
        if not re.match(r"^[A-Z]{1,2}[0-9]{6,7}$", cand) or len(cand) != 8:
            continue
            
        score = data["score"]
        if "front" in data["contexts"] and "back" in data["contexts"]:
            score += 20
            
        if score > best_score:
            best_score = score
            best_cand = cand
            
    return best_cand or ""


def extract_passport_details(
    front_image_path,
    back_image_path=None,
):
    """
    Extract passport information.

    front_image_path:
        Front passport page.

    back_image_path:
        Optional back passport page.
    """

    # --------------------------------------------------------
    # FRONT + BACK OCR (sequential execution to avoid C++ predictor concurrency crash)
    # --------------------------------------------------------
    def _run_ocr(path):
        if not path:
            return ""
        orig, proc = ocr_image(path)
        if orig == proc:
            return orig
        return orig + "\n" + proc

    front_text = _run_ocr(front_image_path)
    back_text = _run_ocr(back_image_path) if back_image_path else ""

    # --------------------------------------------------------
    # MRZ
    # --------------------------------------------------------

    mrz_line1, mrz_line2 = find_mrz_lines(
        front_text
    )

    mrz = parse_mrz(
        mrz_line1,
        mrz_line2,
    )

    # --------------------------------------------------------
    # PASSPORT NUMBER
    # --------------------------------------------------------

    # MRZ is primary truth if valid
    passport_number = ""
    if mrz and mrz.get("passport_number") and re.match(r"^[A-Z][0-9]{7}$", mrz["passport_number"]):
        passport_number = mrz["passport_number"]
    else:
        passport_number = find_best_passport_number(front_text, back_text, mrz)

    # --------------------------------------------------------
    # FRONT FIELDS
    # --------------------------------------------------------

    front_fields = extract_front_fields(
        front_text,
        passport_number
    )

    # --------------------------------------------------------
    # BACK FIELDS
    # --------------------------------------------------------

    back_fields = extract_back_fields(
        back_text
    ) if back_text else {
        "passport_number": "",
        "father_name": "",
        "mother_name": "",
        "spouse_name": "",
        "address": {
            "raw": "",
            "lines": [],
            "pincode": "",
        },
        "file_number": "",
    }

    # --------------------------------------------------------
    # NAMES
    # --------------------------------------------------------

    mrz_surname_clean = remove_mrz_noise(clean_name(mrz["surname"]))
    mrz_given_clean = remove_mrz_noise(clean_name(mrz["given_names"]))

    surname = mrz_surname_clean or front_fields["surname"]
    given_names = mrz_given_clean or front_fields["given_names"]

    surname = clean_name(surname)
    given_names = clean_name(given_names)

    # ---- Back-page name fallback ----
    if (not given_names and not surname) and back_text and passport_number:
        back_lines = [
            normalize_spaces(ln)
            for ln in back_text.splitlines()
            if normalize_spaces(ln)
        ]
        for bi, bline in enumerate(back_lines):
            if passport_number in bline.replace(" ", "") and "<" not in bline:
                for candidate_line in back_lines[bi + 1: bi + 3]:
                    words = re.findall(r'\b[A-Z][A-Z]+\b', candidate_line)
                    ignore = {
                        "EMIGRATION", "CHECK", "REQUIRED", "INDIA", "INDIAN",
                        "PASSPORT", "NAME", "FATHER", "MOTHER", "SPOUSE",
                        "ADDRESS", "FILE", "OLD",
                    }
                    words = [w for w in words if w not in ignore]
                    if len(words) >= 2:
                        given_names = " ".join(words[:-1])
                        surname = words[-1]
                        break
                if given_names or surname:
                    break

    name_parts = []
    if given_names:
        name_parts.append(given_names)
    if surname:
        name_parts.append(surname)

    full_name = " ".join(name_parts)

    # --------------------------------------------------------
    # GENDER / NATIONALITY
    # --------------------------------------------------------

    gender = mrz["gender"] or front_fields["gender"]
    nationality = mrz["nationality"] or front_fields["nationality"]

    # --------------------------------------------------------
    # DOB
    # --------------------------------------------------------

    date_of_birth = mrz["date_of_birth"] or front_fields["date_of_birth"]

    if not date_of_birth:
        dates = extract_dates(front_text)
        if dates:
            date_of_birth = dates[0]

    # --------------------------------------------------------
    # EXPIRY
    # --------------------------------------------------------

    expiry_date = mrz["expiry_date"] or front_fields["expiry_date"]

    # --------------------------------------------------------
    # ISSUE DATE
    # --------------------------------------------------------

    date_of_issue = front_fields["date_of_issue"]

    if not date_of_issue or date_of_issue == date_of_birth:
        dates = extract_dates(front_text)
        non_dob = [d for d in dates if d != date_of_birth]
        if non_dob:
            parsed = []
            for d in non_dob:
                try:
                    parsed.append((datetime.strptime(d, "%d-%m-%Y"), d))
                except ValueError:
                    continue
            parsed.sort(key=lambda x: x[0])
            if parsed:
                date_of_issue = parsed[0][1]

    # --------------------------------------------------------
    # PLACE OF BIRTH / ISSUE
    # --------------------------------------------------------

    place_of_birth = front_fields["place_of_birth"]
    place_of_issue = front_fields["place_of_issue"]

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    filled_fields = 0

    fields_to_check = [
        passport_number,
        full_name,
        gender,
        nationality,
        date_of_birth,
        date_of_issue,
        expiry_date,
        place_of_birth,
        place_of_issue,
        back_fields["father_name"],
        back_fields["mother_name"],
        back_fields["spouse_name"],
        back_fields["address"]["raw"],
    ]

    for field in fields_to_check:
        if field:
            filled_fields += 1

    confidence = (
        "high"
        if filled_fields >= 9
        else "medium"
        if filled_fields >= 5
        else "low"
    )

    return {
        "passport_number": passport_number,
        "name": full_name,
        "surname": surname,
        "given_names": given_names,
        "gender": gender,
        "nationality": nationality,
        "date_of_birth": date_of_birth,
        "date_of_issue": date_of_issue,
        "expiry_date": expiry_date,
        "date_of_expiry": expiry_date,
        "place_of_birth": place_of_birth,
        "place_of_issue": place_of_issue,

        "address": back_fields["address"],
        "city": back_fields.get("city", ""),

        "father_name": back_fields["father_name"],
        "name_of_father": back_fields["father_name"],

        "mother_name": back_fields["mother_name"],
        "name_of_mother": back_fields["mother_name"],

        "spouse_name": back_fields["spouse_name"],

        "file_number": back_fields["file_number"],

        "mrz": {
            "line1": mrz_line1,
            "line2": mrz_line2,
            "valid": mrz["valid"],
        },

        "raw_text_front": front_text,
        "raw_text_back": back_text,

        "extraction_confidence": confidence,
    }


def remove_mrz_noise(value):
    """
    Remove obvious repeated OCR garbage from MRZ names.

    Handles:
      - Single repeated char:  KKKKKKKKK   -> removed
      - Mixed repeated noise:  DASKKRAJUSKKKK -> removed  (>50% repeated char)
      - Alternating garbage:   EKEKKKKK -> removed
    """

    if not value:
        return ""

    words = value.split()
    cleaned = []

    for word in words:
        # All same character: KKKKKK, EEEEE
        if len(word) >= 4 and len(set(word)) == 1:
            continue

        # Word is too long AND dominated by a single repeated char (> 50%)
        if len(word) >= 6:
            from collections import Counter
            counts = Counter(word)
            most_common_char, most_common_count = counts.most_common(1)[0]
            if most_common_count / len(word) > 0.5:
                continue

        # Looks like MRZ filler chars mixed together (alternating E/K pattern)
        if len(word) >= 5:
            unique_chars = set(word.upper())
            mrz_garbage_chars = {'K', 'E', '<'}
            if unique_chars.issubset(mrz_garbage_chars):
                continue

        cleaned.append(word)

    return " ".join(cleaned).strip()


# ============================================================
# PUBLIC API
# ============================================================

def passport(
    front_image_path,
    back_image_path=None,
):
    """
    Public passport extraction API.

    Example:

        passport("front.jpeg")

    or:

        passport(
            "front.jpeg",
            "back.jpeg",
        )
    """

    return extract_passport_details(
        front_image_path,
        back_image_path,
    )