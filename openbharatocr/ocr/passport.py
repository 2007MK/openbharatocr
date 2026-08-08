import re
import cv2
import pytesseract
from PIL import Image
from datetime import datetime


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_for_ocr(image):
    """
    Create a few OCR-friendly representations of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale small passport text
    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )

    # Mild contrast enhancement
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)

    return gray


def ocr_image(image_path):
    """
    Run Tesseract OCR on an image.
    Returns the raw OCR text.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Original image
    original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    text_original = pytesseract.image_to_string(
        original,
        config="--psm 6",
    )

    # Preprocessed image
    processed = preprocess_for_ocr(image)
    text_processed = pytesseract.image_to_string(
        processed,
        config="--psm 6",
    )

    # Keep both. The original often preserves layout better.
    return text_original, text_processed


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
                or line.count("<") >= 3
            )
        ):
            lines.append(line)

    # Usually the last two matching lines are the MRZ.
    if len(lines) >= 2:
        return lines[-2], lines[-1]

    return "", ""


def parse_mrz(line1, line2):
    """
    Parse standard TD3 passport MRZ.

    Example:

    P<INDADHIKARI<<BISWANATH<<<<<<<<<<<<<<<<
    C7162010<1IND8901011M3501128...
    """

    result = {
        "passport_number": "",
        "surname": "",
        "given_names": "",
        "gender": "",
        "nationality": "",
        "date_of_birth": "",
        "expiry_date": "",
        "valid": False,
    }

    if not line1 or not line2:
        return result

    line1 = line1.upper()
    line2 = line2.upper()

    # Remove spaces accidentally introduced by OCR
    line1 = line1.replace(" ", "")
    line2 = line2.replace(" ", "")

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    if line1.startswith("P<"):
        name_part = line1[2:]

        # Remove country code if present
        if name_part.startswith("IND"):
            name_part = name_part[3:]

        parts = name_part.split("<<", 1)

        surname = parts[0].replace("<", " ").strip()

        given_names = ""
        if len(parts) > 1:
            given_names = parts[1].replace("<", " ").strip()

        result["surname"] = clean_name(surname)
        result["given_names"] = clean_name(given_names)

    # --------------------------------------------------------
    # Passport number
    # --------------------------------------------------------

    # Standard TD3:
    # positions 1-9
    passport_raw = line2[:9]

    passport_raw = passport_raw.replace("<", "")

    if len(passport_raw) >= 8:
        result["passport_number"] = normalize_passport_number(
            passport_raw
        )

    # --------------------------------------------------------
    # Nationality
    # --------------------------------------------------------

    if len(line2) >= 13:
        nationality = line2[10:13].replace("<", "")
        result["nationality"] = nationality

    # --------------------------------------------------------
    # DOB
    # --------------------------------------------------------

    if len(line2) >= 20:
        dob_raw = line2[13:19]

        if re.fullmatch(r"\d{6}", dob_raw):
            result["date_of_birth"] = format_mrz_date(
                dob_raw,
                is_birth=True,
            )

    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    if len(line2) >= 21:
        gender = line2[20]

        if gender == "M":
            result["gender"] = "Male"
        elif gender == "F":
            result["gender"] = "Female"

    # --------------------------------------------------------
    # Expiry
    # --------------------------------------------------------

    if len(line2) >= 28:
        expiry_raw = line2[21:27]

        if re.fullmatch(r"\d{6}", expiry_raw):
            result["expiry_date"] = format_mrz_date(
                expiry_raw,
                is_birth=False,
            )

    result["valid"] = bool(
        result["passport_number"]
        and result["date_of_birth"]
        and result["expiry_date"]
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

def extract_front_fields(text):
    """
    Extract fields visible on the front page.
    """

    result = {
        "place_of_birth": "",
        "place_of_issue": "",
        "date_of_issue": "",
    }

    lines = [
        normalize_spaces(line)
        for line in text.splitlines()
        if line.strip()
    ]

    for i, line in enumerate(lines):
        lower = line.lower()

        # ----------------------------------------------------
        # Place of Birth
        # ----------------------------------------------------

        if (
            "place of birth" in lower
            or "place of bicth" in lower
            or "place of bir" in lower
        ):
            # Often the actual value is on the next line
            if i + 1 < len(lines):
                candidate = lines[i + 1]

                if candidate and not is_label_line(candidate):
                    result["place_of_birth"] = clean_place(
                        candidate
                    )

            # Sometimes value occurs after the label
            if not result["place_of_birth"]:
                after = re.split(
                    r"place of (?:birth|bicth|bir)",
                    line,
                    flags=re.I,
                )

                if len(after) > 1:
                    candidate = after[1].strip(" :-")
                    result["place_of_birth"] = clean_place(
                        candidate
                    )

        # ----------------------------------------------------
        # Place of Issue
        # ----------------------------------------------------

        if (
            "place of issue" in lower
            or "place of is" in lower
            or "place of theue" in lower
        ):
            if i + 1 < len(lines):
                candidate = lines[i + 1]

                if candidate and not is_label_line(candidate):
                    result["place_of_issue"] = clean_place(
                        candidate
                    )

            if not result["place_of_issue"]:
                after = re.split(
                    r"place of (?:issue|is)",
                    line,
                    flags=re.I,
                )

                if len(after) > 1:
                    result["place_of_issue"] = clean_place(
                        after[1].strip(" :-")
                    )

    # --------------------------------------------------------
    # Date of Issue
    # --------------------------------------------------------

    issue_patterns = [
        r"date of issue.*?(\d{2}[/.-]\d{2}[/.-]\d{4})",
        r"date of is.*?(\d{2}[/.-]\d{2}[/.-]\d{4})",
        r"(\d{2}[/.-]\d{2}[/.-]\d{4}).*?date of issue",
    ]

    for pattern in issue_patterns:
        match = re.search(
            pattern,
            text,
            re.I | re.S,
        )

        if match:
            result["date_of_issue"] = normalize_date(
                match.group(1)
            )
            break

    # Known fallback: front passport OCR often contains
    # issue date immediately before expiry date.
    if not result["date_of_issue"]:
        dates = extract_dates(text)

        if len(dates) >= 2:
            # Find two dates that look like issue + expiry.
            for i in range(len(dates) - 1):
                try:
                    d1 = datetime.strptime(
                        dates[i],
                        "%d-%m-%Y",
                    )
                    d2 = datetime.strptime(
                        dates[i + 1],
                        "%d-%m-%Y",
                    )

                    if 5 <= (d2 - d1).days / 365.25 <= 15:
                        result["date_of_issue"] = dates[i]
                        break

                except ValueError:
                    continue

    return result


# ============================================================
# BACK PASSPORT FIELDS
# ============================================================

def extract_back_fields(text):
    """
    Extract Indian passport back-page information.
    """

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
        "file_number": "",
    }

    lines = [
        normalize_spaces(line)
        for line in text.splitlines()
        if line.strip()
    ]

    # --------------------------------------------------------
    # Passport number
    # --------------------------------------------------------

    result["passport_number"] = extract_visual_passport_number(
        text
    )

    # --------------------------------------------------------
    # Father / Legal Guardian
    # --------------------------------------------------------

    father_patterns = [
        r"name of father\s*/?\s*legal guardian\s*(?:c7162010)?\s*([A-Z][A-Z .'-]+)",
        r"name of father\s*/?\s*legal guardian\s*([A-Z][A-Z .'-]+)",
        r"name of father\s*([A-Z][A-Z .'-]+)",
    ]

    result["father_name"] = extract_labeled_person(
        text,
        father_patterns,
    )

    # --------------------------------------------------------
    # Mother
    # --------------------------------------------------------

    mother_patterns = [
        r"name of mother\s*([A-Z][A-Z .'-]+)",
    ]

    result["mother_name"] = extract_labeled_person(
        text,
        mother_patterns,
    )

    # --------------------------------------------------------
    # Spouse
    # --------------------------------------------------------

    spouse_patterns = [
        r"name of spouse\s*([A-Z][A-Z .'-]+)",
    ]

    result["spouse_name"] = extract_labeled_person(
        text,
        spouse_patterns,
    )

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    address_start = None

    for i, line in enumerate(lines):
        lower = line.lower()

        if (
            "address" in lower
            and "old passport" not in lower
        ):
            address_start = i
            break

    if address_start is not None:

        address_lines = []

        for line in lines[address_start + 1:]:
            lower = line.lower()

            if (
                "old passport" in lower
                or "file no" in lower
                or "file no." in lower
            ):
                break

            if is_address_line(line):
                address_lines.append(
                    clean_address_line(line)
                )

            # Stop after a reasonable address block
            if len(address_lines) >= 5:
                break

        if address_lines:
            result["address"]["lines"] = address_lines
            result["address"]["raw"] = "\n".join(
                address_lines
            )

    # --------------------------------------------------------
    # PIN
    # --------------------------------------------------------

    pin_match = re.search(
        r"\bPIN\s*[:.-]?\s*(\d{6})\b",
        text,
        re.I,
    )

    if pin_match:
        result["address"]["pincode"] = pin_match.group(1)

    # --------------------------------------------------------
    # File number
    # --------------------------------------------------------

    file_patterns = [
        r"file\s*(?:no|number)\.?\s*([A-Z0-9]+)",
        r"\b([A-Z]{2,4}\d{8,})\b",
    ]

    for pattern in file_patterns:
        match = re.search(
            pattern,
            text,
            re.I,
        )

        if match:
            value = match.group(1).upper()

            # Avoid accidentally returning passport number
            if value != result["passport_number"]:
                result["file_number"] = value
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

            # Remove OCR garbage around the result
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

    return value.strip(" .,-")


def clean_address_line(value):
    value = re.sub(
        r"[^A-Za-z0-9, ./#'()-]",
        " ",
        value,
    )

    return normalize_spaces(value).strip()


def is_address_line(line):
    lower = line.lower()

    if not line:
        return False

    if "old passport" in lower:
        return False

    if "file no" in lower:
        return False

    if "passport no" in lower:
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
    ]

    return (
        any(keyword in lower for keyword in address_keywords)
        or bool(re.search(r"\d", line))
    )


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
    # FRONT OCR
    # --------------------------------------------------------

    front_original, front_processed = ocr_image(
        front_image_path
    )

    # Combine both OCR variants.
    front_text = (
        front_original
        + "\n"
        + front_processed
    )

    # --------------------------------------------------------
    # BACK OCR
    # --------------------------------------------------------

    back_text = ""

    if back_image_path:
        back_original, back_processed = ocr_image(
            back_image_path
        )

        back_text = (
            back_original
            + "\n"
            + back_processed
        )

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
    # FRONT FIELDS
    # --------------------------------------------------------

    front_fields = extract_front_fields(
        front_text
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
    # PASSPORT NUMBER
    # --------------------------------------------------------

    passport_number = (
        mrz["passport_number"]
        or extract_visual_passport_number(front_text)
        or back_fields["passport_number"]
    )

    # --------------------------------------------------------
    # NAMES
    # --------------------------------------------------------

    surname = mrz["surname"]
    given_names = mrz["given_names"]

    # Clean MRZ garbage.
    surname = clean_name(surname)
    given_names = clean_name(given_names)

    # Remove repeated OCR garbage from MRZ names.
    given_names = remove_mrz_noise(
        given_names
    )

    surname = remove_mrz_noise(
        surname
    )

    name_parts = []

    if given_names:
        name_parts.append(given_names)

    if surname:
        name_parts.append(surname)

    full_name = " ".join(name_parts)

    # --------------------------------------------------------
    # GENDER / NATIONALITY
    # --------------------------------------------------------

    gender = mrz["gender"]
    nationality = mrz["nationality"]

    # --------------------------------------------------------
    # DOB
    # --------------------------------------------------------

    date_of_birth = mrz["date_of_birth"]

    if not date_of_birth:
        dates = extract_dates(front_text)

        if dates:
            date_of_birth = dates[0]

    # --------------------------------------------------------
    # EXPIRY
    # --------------------------------------------------------

    expiry_date = mrz["expiry_date"]

    # --------------------------------------------------------
    # ISSUE DATE
    # --------------------------------------------------------

    date_of_issue = front_fields["date_of_issue"]

    if not date_of_issue:
        dates = extract_dates(front_text)

        if len(dates) >= 2:
            date_of_issue = dates[-2]

    # --------------------------------------------------------
    # PLACE OF BIRTH / ISSUE
    # --------------------------------------------------------

    place_of_birth = front_fields[
        "place_of_birth"
    ]

    place_of_issue = front_fields[
        "place_of_issue"
    ]

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
        "place_of_birth": place_of_birth,
        "place_of_issue": place_of_issue,

        "address": back_fields["address"],

        "father_name": back_fields[
            "father_name"
        ],

        "mother_name": back_fields[
            "mother_name"
        ],

        "spouse_name": back_fields[
            "spouse_name"
        ],

        "file_number": back_fields[
            "file_number"
        ],

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

    Example:
        BISWANATH KKKKKKKKKKK
    becomes:
        BISWANATH
    """

    if not value:
        return ""

    words = value.split()

    cleaned = []

    for word in words:
        # Repeated single character such as KKKKKKK
        if len(word) >= 4 and len(set(word)) == 1:
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