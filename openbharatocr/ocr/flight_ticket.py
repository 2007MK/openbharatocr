import re
import datetime
import os

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz
        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False

# -----------------------------------------------------------------------
# IATA airline code -> human-readable airline name
# Covers all major Indian airlines + top international carriers
# -----------------------------------------------------------------------
AIRLINE_NAMES = {
    # Indian carriers
    "6E": "IndiGo", "AI": "Air India", "SG": "SpiceJet",
    "G8": "Go First", "IX": "Air India Express", "UK": "Vistara",
    "I5": "Air Asia India", "QP": "Akasa Air",
    # International carriers commonly used from India
    "EK": "Emirates", "QR": "Qatar Airways", "EY": "Etihad Airways",
    "SQ": "Singapore Airlines", "MH": "Malaysia Airlines",
    "TG": "Thai Airways", "BA": "British Airways", "LH": "Lufthansa",
    "AF": "Air France", "KL": "KLM", "TK": "Turkish Airlines",
    "CX": "Cathay Pacific", "NH": "ANA", "JL": "Japan Airlines",
    "QF": "Qantas", "UA": "United Airlines", "AA": "American Airlines",
    "DL": "Delta Air Lines", "WS": "WestJet", "AC": "Air Canada",
    "FR": "Ryanair", "U2": "easyJet", "W6": "Wizz Air",
    "FZ": "flydubai", "G9": "Air Arabia", "XY": "flynas",
    "PK": "Pakistan International Airlines", "UL": "SriLankan Airlines",
    "BG": "Biman Bangladesh Airlines", "RA": "Nepal Airlines",
    "FJ": "Fiji Airways", "WY": "Oman Air", "GF": "Gulf Air",
    "KE": "Korean Air", "OZ": "Asiana Airlines",
    "CA": "Air China", "MU": "China Eastern", "CZ": "China Southern",
    "ET": "Ethiopian Airlines", "KQ": "Kenya Airways",
    "MS": "EgyptAir", "RJ": "Royal Jordanian", "ME": "Middle East Airlines",
    "SV": "Saudia", "WB": "Rwandair",
    # AirAsia group
    "FD": "Thai AirAsia", "AK": "AirAsia", "QZ": "Indonesia AirAsia",
    "Z2": "Philippines AirAsia", "D7": "AirAsia X",
    # Other regional
    "XT": "Indonesia AirAsia X", "XJ": "Thai AirAsia X",
    "BT": "airBaltic", "LO": "LOT Polish Airlines",
    "OS": "Austrian Airlines", "LX": "Swiss International",
    "SK": "SAS", "AY": "Finnair", "TP": "TAP Air Portugal",
}

# -----------------------------------------------------------------------
# IATA airport code -> city name
# -----------------------------------------------------------------------
IATA_CITY = {
    # India
    "BLR": "Bengaluru", "DEL": "Delhi", "BOM": "Mumbai", "MAA": "Chennai",
    "CCU": "Kolkata", "HYD": "Hyderabad", "COK": "Kochi", "AMD": "Ahmedabad",
    "GOI": "Goa", "PNQ": "Pune", "JAI": "Jaipur", "LKO": "Lucknow",
    "ATQ": "Amritsar", "IXC": "Chandigarh", "BBI": "Bhubaneswar",
    "VTZ": "Visakhapatnam", "TRV": "Thiruvananthapuram", "IXZ": "Port Blair",
    # SE Asia / Pacific
    "BKK": "Bangkok", "DMK": "Bangkok", "CNX": "Chiang Mai", "HKT": "Phuket",
    "SIN": "Singapore", "KUL": "Kuala Lumpur", "CGK": "Jakarta",
    "DPS": "Bali", "MNL": "Manila", "SGN": "Ho Chi Minh City",
    "HAN": "Hanoi", "REP": "Siem Reap", "RGN": "Yangon",
    "PNH": "Phnom Penh", "VTE": "Vientiane",
    # East Asia
    "NRT": "Tokyo", "HND": "Tokyo", "KIX": "Osaka", "NGO": "Nagoya",
    "ICN": "Seoul", "GMP": "Seoul", "PEK": "Beijing", "PVG": "Shanghai",
    "HKG": "Hong Kong", "TPE": "Taipei",
    # Middle East
    "DXB": "Dubai", "AUH": "Abu Dhabi", "SHJ": "Sharjah",
    "DOH": "Doha", "BAH": "Bahrain", "KWI": "Kuwait",
    "MCT": "Muscat", "RUH": "Riyadh", "JED": "Jeddah",
    "AMM": "Amman", "IST": "Istanbul", "SAW": "Istanbul",
    # Europe
    "WAW": "Warsaw", "VIE": "Vienna", "ZRH": "Zurich", "PRG": "Prague",
    "BUD": "Budapest", "CPH": "Copenhagen", "HEL": "Helsinki", "OSL": "Oslo",
    "ARN": "Stockholm", "DUB": "Dublin", "LIS": "Lisbon", "ATH": "Athens",
    "BER": "Berlin", "MXP": "Milan", "LHR": "London", "LGW": "London",
    "STN": "London", "MAN": "Manchester", "EDI": "Edinburgh",
    "CDG": "Paris", "ORY": "Paris", "FRA": "Frankfurt",
    "MUC": "Munich", "AMS": "Amsterdam", "FCO": "Rome",
    "BCN": "Barcelona", "MAD": "Madrid", "BRU": "Brussels", "SVO": "Moscow",
    # Americas
    "JFK": "New York", "EWR": "New York", "LGA": "New York",
    "LAX": "Los Angeles", "ORD": "Chicago", "SFO": "San Francisco",
    "IAD": "Washington", "BOS": "Boston", "YYZ": "Toronto", "YVR": "Vancouver",
    # Africa / Oceania
    "CAI": "Cairo", "NBO": "Nairobi", "ADD": "Addis Ababa", "JNB": "Johannesburg",
    "SYD": "Sydney", "MEL": "Melbourne", "BNE": "Brisbane", "PER": "Perth",
    "AKL": "Auckland",
    # Sri Lanka / Maldives / Nepal / Bangladesh
    "CMB": "Colombo", "MLE": "Malé", "KTM": "Kathmandu",
    "DAC": "Dhaka",
}

CITY_COUNTRY_MAP = {
    "bengaluru": "India", "delhi": "India", "mumbai": "India", "chennai": "India",
    "kolkata": "India", "hyderabad": "India", "kochi": "India", "ahmedabad": "India",
    "goa": "India", "pune": "India", "jaipur": "India", "lucknow": "India",
    "bangkok": "Thailand", "phuket": "Thailand", "singapore": "Singapore",
    "kuala lumpur": "Malaysia", "tokyo": "Japan", "osaka": "Japan",
    "seoul": "South Korea", "beijing": "China", "shanghai": "China",
    "hong kong": "Hong Kong", "taipei": "Taiwan", "dubai": "UAE",
    "abu dhabi": "UAE", "sharjah": "UAE", "doha": "Qatar", "bahrain": "Bahrain",
    "kuwait": "Kuwait", "muscat": "Oman", "riyadh": "Saudi Arabia",
    "jeddah": "Saudi Arabia", "istanbul": "Turkey", "london": "UK",
    "manchester": "UK", "edinburgh": "UK", "heathrow": "UK", "paris": "France",
    "frankfurt": "Germany", "munich": "Germany", "berlin": "Germany",
    "amsterdam": "Netherlands", "rome": "Italy", "milan": "Italy",
    "barcelona": "Spain", "madrid": "Spain", "zurich": "Switzerland",
    "vienna": "Austria", "prague": "Czech Republic", "budapest": "Hungary",
    "copenhagen": "Denmark", "helsinki": "Finland", "oslo": "Norway",
    "stockholm": "Sweden", "dublin": "Ireland", "lisbon": "Portugal",
    "athens": "Greece", "warsaw": "Poland", "brussels": "Belgium",
    "moscow": "Russia", "new york": "USA", "los angeles": "USA",
    "chicago": "USA", "san francisco": "USA", "washington": "USA",
    "boston": "USA", "toronto": "Canada", "vancouver": "Canada",
    "sydney": "Australia", "melbourne": "Australia", "brisbane": "Australia",
    "perth": "Australia", "auckland": "New Zealand", "colombo": "Sri Lanka",
    "malé": "Maldives", "male": "Maldives", "kathmandu": "Nepal",
    "dhaka": "Bangladesh", "cairo": "Egypt",
}

COUNTRY_KEYWORDS = {
    "united kingdom": "UNITED KINGDOM",
    "uk": "UK",
    "india": "India",
    "dubai": "UAE",
    "uae": "UAE",
    "united arab emirates": "UAE",
    "abu dhabi": "UAE",
    "sharjah": "UAE",
    "singapore": "Singapore",
    "malaysia": "Malaysia",
    "kuala lumpur": "Malaysia",
    "thailand": "Thailand",
    "bangkok": "Thailand",
    "japan": "Japan",
    "south korea": "South Korea",
    "korea": "South Korea",
    "china": "China",
    "hong kong": "Hong Kong",
    "taiwan": "Taiwan",
    "sri lanka": "Sri Lanka",
    "nepal": "Nepal",
    "maldives": "Maldives",
    "qatar": "Qatar",
    "doha": "Qatar",
    "bahrain": "Bahrain",
    "kuwait": "Kuwait",
    "oman": "Oman",
    "saudi arabia": "Saudi Arabia",
    "saudi": "Saudi Arabia",
    "jordan": "Jordan",
    "turkey": "Turkey",
    "france": "France",
    "germany": "Germany",
    "netherlands": "Netherlands",
    "italy": "Italy",
    "spain": "Spain",
    "switzerland": "Switzerland",
    "austria": "Austria",
    "poland": "Poland",
    "czech republic": "Czech Republic",
    "hungary": "Hungary",
    "denmark": "Denmark",
    "finland": "Finland",
    "norway": "Norway",
    "sweden": "Sweden",
    "ireland": "Ireland",
    "portugal": "Portugal",
    "greece": "Greece",
    "egypt": "Egypt",
    "belgium": "Belgium",
    "russia": "Russia",
    "usa": "USA",
    "united states": "USA",
    "canada": "Canada",
    "brazil": "Brazil",
    "kenya": "Kenya",
    "ethiopia": "Ethiopia",
    "south africa": "South Africa",
    "australia": "Australia",
    "new zealand": "New Zealand",
}


def _score_ticket_page(page_text: str) -> int:
    if not page_text:
        return 0
    text_lower = page_text.lower()
    pos_keywords = [
        "electronic ticket", "e-ticket", "ticket receipt", "itinerary",
        "boarding pass", "booking reference", "etihad reference", "pnr",
        "flight details", "passenger name", "flight", "airline",
        "departure", "arrival", "terminal", "gate", "seat", "class",
        "baggage", "fare type", "confirmed", "airways", "air india",
        "indigo", "spicejet", "vistara", "emirates", "qatar"
    ]
    neg_keywords = [
        "lrs form", "form a2", "fx sale", "cash memo", "currency sale",
        "tax invoice", "declaration under section", "prove your status",
        "british-irish visa", "bank receipt", "bill no"
    ]
    score = 0
    for kw in pos_keywords:
        if kw in text_lower:
            score += 1
    for kw in neg_keywords:
        if kw in text_lower:
            score -= 2
    return score


def _is_text_garbled(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return False
    # Check printable ASCII ratio
    printable_count = sum(1 for c in text if 32 <= ord(c) <= 126 or c in '\n\r\t')
    ratio = printable_count / len(text)
    if ratio < 0.70:
        return True
    
    # Check common flight ticket keywords
    text_lower = text.lower()
    common_keywords = [
        "ticket", "flight", "pnr", "booking", "passenger", "itinerary",
        "airline", "airways", "departure", "arrival", "date", "reference",
        "receipt", "confirmed", "terminal", "seat", "class", "air"
    ]
    kw_matches = sum(1 for kw in common_keywords if kw in text_lower)
    if len(text.strip()) > 80 and kw_matches == 0:
        return True
    return False


def _ocr_image_file(image_path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        res = pytesseract.image_to_string(img)
        if res and res.strip():
            return res
    except Exception:
        pass

    try:
        from openbharatocr.core.engine import extract_text_paddle
        return extract_text_paddle(image_path)
    except Exception:
        pass
    return ""


def _ocr_pdf_page(page, dpi=200) -> str:
    try:
        import pytesseract
        from PIL import Image
        import io
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        res = pytesseract.image_to_string(img)
        if res and res.strip():
            return res
    except Exception:
        pass

    try:
        from openbharatocr.core.engine import extract_text_paddle
        import tempfile
        pix = page.get_pixmap(dpi=dpi)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            pix.save(tmp_path)
        try:
            return extract_text_paddle(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        pass
    return ""


def extract_ticket_details(pdf_path):
    """
    Extract flight ticket details from a PDF, text, or image file.
    """
    if not pdf_path:
        return extract_details_from_text("")

    text = ""
    # 1. Text file handling
    if isinstance(pdf_path, str) and pdf_path.lower().endswith('.txt'):
        try:
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return extract_details_from_text(text)
        except Exception as e:
            return {"error": f"Failed to read text file: {str(e)}"}

    # 2. Image file handling
    is_img = False
    if isinstance(pdf_path, str):
        ext = pdf_path.lower().split('.')[-1]
        if ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
            is_img = True

    if is_img:
        text = _ocr_image_file(pdf_path)
        return extract_details_from_text(text)

    # 3. PDF handling
    if not HAS_PYMUPDF:
        text = _ocr_image_file(pdf_path)
        if text:
            return extract_details_from_text(text)
        raise ImportError("PyMuPDF is required for PDF parsing.")

    doc = None
    try:
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
        except ImportError:
            import fitz
            doc = fitz.open(pdf_path)

        page_scores = []
        page_texts = []
        for page in doc:
            p_text = page.get_text("text")
            # If text is garbled or empty, run OCR for scoring accuracy
            if _is_text_garbled(p_text) or not p_text.strip():
                ocr_p = _ocr_pdf_page(page)
                if ocr_p.strip():
                    p_text = ocr_p
            page_texts.append(p_text)
            page_scores.append(_score_ticket_page(p_text))

        max_score = max(page_scores) if page_scores else 0

        selected_pages = []
        if max_score > 0:
            for idx, score in enumerate(page_scores):
                if score > 0:
                    selected_pages.append(idx)
        else:
            selected_pages = list(range(len(doc)))

        extracted_parts = [page_texts[idx] for idx in selected_pages]
        text = "\n\n".join(extracted_parts)

    except Exception as e:
        if doc:
            doc.close()
        return {"error": f"Failed to read PDF: {str(e)}"}

    if doc:
        doc.close()

    result = extract_details_from_text(text)
    return result


def extract_details_from_text(text):
    if not text:
        text = ""

    result = {
        "pnr": "",
        "passenger_names": [],
        "airline_name": "",
        "flight_number": "",
        "source": "",
        "destination_country": "",
        "departure_date": "",
        "return_date": "",
        "duration": "",
        # Backward compatibility aliases
        "pnr_number": "",
        "travellers": [],
        "airline_names": [],
        "flight_details": [],
        "origin": {"city": "", "country": ""},
        "destination": {"city": "", "country": ""},
        "onward_date": "",
    }

    # 1. PNR extraction
    pnr_val = ""
    pnr_blacklist = {
        "FLIGHT", "TICKET", "STATUS", "CANCEL", "AUGUST", "ISSUED",
        "PASSEX", "AMOUNT", "CHARGE", "LUGGAGE", "GUEST", "DEPART",
        "ARRIVE", "SINGLE", "DOUBLE", "MASTER", "SELECT", "MEMBER",
        "PERSON", "DINNER", "SECTOR", "CABIN", "RECORD", "NUMBER",
        "DETAIL", "NOTICE", "TRAVEL", "EXPERT", "ACTION", "ROUTING",
        "SYSTEM", "POLICY", "PORTAL", "CHANGE", "REFUND", "ONLINE"
    }
    e_ticket_numbers = set(re.findall(r'\b\d{13}\b', text))
    e_ticket_numbers.update(re.findall(r'\b\d{3}[\s-]?\d{10}\b', text))
    e_ticket_numbers.update(re.findall(r'\b\d{10,12}\b', text))

    pnr_match = re.search(
        r'(?i:\b(?:PNR|Booking\s*(?:Ref(?:erence)?|Code|No|ID|Number)|Airline\s*PNR|Etihad\s*reference|Record\s*Locator|Reservation\s*(?:Code|Number|Ref)|PNR\s*Code|PNR\s*No|Loc)\b)[\s:-]*(?:[A-Z0-9]{2}/)?([A-Z0-9]{6})\b',
        text
    )
    if not pnr_match:
        pnr_match = re.search(
            r'(?i:\b(?:Booking|Ref|Reference|Confirmation)\b)[\s:-]*(?:[A-Z0-9]{2}/)?([A-Z0-9]{6})\b',
            text
        )

    if pnr_match:
        cand = pnr_match.group(1).upper()
        if len(cand) == 6 and cand not in pnr_blacklist and not cand.isdigit():
            pnr_val = cand

    if not pnr_val:
        pnr_line_match = re.search(r'(?i:\bPNR\b)[\s:-]*([A-Z0-9]{6,10})', text)
        if pnr_line_match:
            cand = pnr_line_match.group(1).upper()
            cand_6 = cand[:6]
            if not cand_6.isdigit() and cand_6 not in pnr_blacklist:
                pnr_val = cand_6

    if not pnr_val:
        for m in re.finditer(r'\b([A-Z0-9]{6})\b', text):
            cand = m.group(1).upper()
            if not cand.isdigit() and cand not in pnr_blacklist:
                if not any(cand in et for et in e_ticket_numbers):
                    pnr_val = cand
                    break

    result["pnr"] = pnr_val
    result["pnr_number"] = pnr_val

    # 2. Passenger Names extraction
    passenger_list = []
    PASSENGER_BLACKLIST = {
        "BREAKFAST", "DINNER", "LUNCH", "MEAL", "PERSON", "KG", "ETKT", "TICKET",
        "SECTOR", "NONSTOP", "STOP", "ECONOMY", "BUSINESS", "STANDARD", "FLEX",
        "VALUE", "CLASS", "FLIGHT", "ALLOWANCE", "CHECKED", "CABIN", "BAGGAGE",
        "EQUIPMENT", "EMISSIONS", "CALCULATOR", "NOTICE", "GENERAL", "INFORMATION",
        "STATUS", "SERVICE", "CHARGES", "TAX", "TOTAL", "FARE", "RULES", "CO2",
        "PASSENGER", "TRAVELLER", "GUEST", "SUMMARY", "DETAILS", "BOOKING", "REFERENCE",
        "RECORD", "LOCATOR", "DEPARTURE", "ARRIVAL", "TERMINAL", "GATE", "SEAT",
        "AIRLINE", "CARRIER", "OPERATING", "CONFIRMED", "ISSUED", "DATE", "TIME"
    }

    label_matches = re.finditer(
        r"(?i:\b(?:Passenger\s*Name|Passenger|Traveller|Guest\s*Name|Pax\s*Name|Customer\s*Name)\b)[\s:-]*([A-Za-z\s.,/\'-]{3,50})",
        text
    )
    for lm in label_matches:
        raw_val = lm.group(1).strip()
        raw_val = raw_val.split('\n')[0].strip()
        raw_val = re.sub(r'(?i)\b(?:PNR|Flight|Date|Ticket|Booking|Airline|Source|Destination|Duration)\b.*$', '', raw_val).strip()
        if raw_val:
            passenger_list.append(raw_val)

    for tm in re.finditer(
        r'\b(Mr|Mrs|Ms|Miss|Mstr|Dr)\.?[ \t]+([A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+)*)\b',
        text, re.IGNORECASE
    ):
        title = tm.group(1).capitalize()
        raw_name = tm.group(2)
        formatted = " ".join(p.title() for p in raw_name.split())
        full_name = f"{title} {formatted}"
        passenger_list.append(full_name)

    # Strategy C: IATA slash format "DARLA/DEEPTHI MRS" or "UMARANI/GANGADHAR SHANKAR MR"
    for sm in re.finditer(
        r'\b([A-Z]{2,25})/([A-Z]+(?:[ \t]+[A-Z]+)*)(?:\s+(MR|MRS|MS|MISS|MSTR|DR))?\b',
        text
    ):
        surname_raw = sm.group(1).upper()
        given_raw = sm.group(2).upper()
        title_raw = (sm.group(3) or "").capitalize() if sm.lastindex and sm.lastindex >= 3 else ""

        given_words = given_raw.split()
        if given_words and given_words[-1] in {"MR", "MRS", "MS", "MISS", "MSTR", "DR"}:
            if not title_raw:
                title_raw = given_words[-1].capitalize()
            given_words = given_words[:-1]

        if surname_raw in PASSENGER_BLACKLIST or surname_raw in IATA_CITY:
            continue
        if not given_words or any(w in PASSENGER_BLACKLIST or w in IATA_CITY for w in given_words):
            continue

        surname = surname_raw.title()
        given = " ".join(w.title() for w in given_words)
        full_name = f"{title_raw} {given} {surname}".strip()
        passenger_list.append(full_name)

    cleaned_passengers = []
    forbidden_words = {"Passenger Name", "Flight Details", "Booking Reference", "Etihad Reference", "Date Of Travel"}
    for p in passenger_list:
        p_clean = p.strip()
        if not p_clean or p_clean in forbidden_words:
            continue
        p_lower = p_clean.lower()
        if not any(p_lower == existing.lower() for existing in cleaned_passengers):
            cleaned_passengers.append(p_clean)

    final_passengers = []
    for p in cleaned_passengers:
        p_lower = p.lower()
        p_core = re.sub(r'^(mr|mrs|ms|miss|mstr|dr)\.?\s*', '', p_lower).strip()
        is_subset = False
        for other in cleaned_passengers:
            if p.lower() == other.lower():
                continue
            other_core = re.sub(r'^(mr|mrs|ms|miss|mstr|dr)\.?\s*', '', other.lower()).strip()
            if p_core in other_core and len(p_core) < len(other_core):
                is_subset = True
                break
            if p_core == other_core and p.lower().startswith('mr ') and other.lower().startswith('mrs '):
                is_subset = True
                break
        if not is_subset:
            final_passengers.append(p)

    result["passenger_names"] = final_passengers
    result["travellers"] = final_passengers

    # 3. Flight number & Airline name extraction
    seen_flight_numbers = []
    seen_airlines = []

    AIRLINE_HEADER_WORDS = {
        "NAME", "AIRLINE", "AIRLINES", "CARRIER", "FLIGHT", "CODE", "DETAILS",
        "SUMMARY", "INFORMATION", "STATUS", "TYPE", "NUMBER", "NO", "NAME(S)"
    }

    airline_match = re.search(
        r'(?i:\b(?:Airline|Carrier|Operating\s*Airline)\b)[\s:-]*([A-Za-z0-9\s]{2,30})',
        text
    )
    if airline_match:
        val = airline_match.group(1).strip().split('\n')[0].strip()
        val = re.sub(r'(?i)\b(?:Source|Destination|Flight|Departure|Date)\b.*$', '', val).strip()
        if val and val.upper() not in AIRLINE_HEADER_WORDS and not val.upper().startswith("NAME") and val not in seen_airlines:
            seen_airlines.append(val)

    flight_match = re.search(
        r'(?i:\b(?:Flight\s*(?:No|Number|Code)?|Flight)\b)[\s:-]*([A-Z0-9]{2,3}[\s-]*\d{1,4})\b',
        text
    )
    if flight_match:
        fl_val = flight_match.group(1).strip().upper()
        if fl_val not in seen_flight_numbers:
            seen_flight_numbers.append(fl_val)

    for match in re.finditer(r'\b([A-Z0-9]{2})[\s-]*(\d{1,4})\b', text):
        code = match.group(1).upper()
        if code in AIRLINE_NAMES:
            fl_code = f"{code} {match.group(2)}"
            if fl_code not in seen_flight_numbers:
                seen_flight_numbers.append(fl_code)
            air_name = AIRLINE_NAMES[code]
            if air_name not in seen_airlines:
                seen_airlines.insert(0, air_name)

    text_lower = text.lower()
    for name in AIRLINE_NAMES.values():
        if name.lower() in text_lower and name not in seen_airlines:
            if name.lower() == "emirates" and "united arab emirates" in text_lower:
                if "ek" in [f.split()[0].lower() for f in seen_flight_numbers] or "emirates airlines" in text_lower:
                    seen_airlines.append(name)
            else:
                seen_airlines.append(name)

    EXTRA_AIRLINE_NAMES = [
        "IndiGo", "Air India", "SpiceJet", "Go First", "GoFirst",
        "Vistara", "Akasa Air", "Air Asia", "AirAsia",
        "Emirates", "Qatar Airways", "Etihad", "Etihad Airways",
        "Singapore Airlines", "Malaysia Airlines",
        "British Airways", "Lufthansa", "Air France", "KLM",
        "Turkish Airlines", "Cathay Pacific",
        "United Airlines", "Delta Air Lines", "American Airlines",
        "flydubai", "Air Arabia", "SriLankan Airlines", "LOT Polish Airlines",
    ]
    for name in EXTRA_AIRLINE_NAMES:
        if name.lower() in text_lower and name not in seen_airlines:
            if name.lower() == "emirates" and "united arab emirates" in text_lower:
                if "ek" in [f.split()[0].lower() for f in seen_flight_numbers] or "emirates airlines" in text_lower:
                    seen_airlines.append(name)
            else:
                seen_airlines.append(name)

    clean_seen_airlines = []
    for a in seen_airlines:
        if a.upper() in AIRLINE_HEADER_WORDS or a.upper().startswith("NAME"):
            continue
        if not any(a.lower() == existing.lower() for existing in clean_seen_airlines):
            clean_seen_airlines.append(a)

    if len(clean_seen_airlines) > 1:
        to_remove = []
        for a in clean_seen_airlines:
            for b in clean_seen_airlines:
                if a != b and a.lower() in b.lower():
                    to_remove.append(a)
                    break
        clean_seen_airlines = [a for a in clean_seen_airlines if a not in to_remove]

    result["airline_name"] = clean_seen_airlines[0] if clean_seen_airlines else ""
    result["airline_names"] = clean_seen_airlines

    result["flight_number"] = seen_flight_numbers[0] if seen_flight_numbers else ""
    result["flight_details"] = seen_flight_numbers

    # 4. Source & Destination Country extraction
    source_val = ""
    dest_country_val = ""
    dest_city_val = ""

    all_iata = [c for c in re.findall(r'\b([A-Z]{3})\b', text) if c in IATA_CITY]
    unique_iata = []
    for c in all_iata:
        if c not in unique_iata:
            unique_iata.append(c)

    if len(unique_iata) >= 1:
        source_val = IATA_CITY[unique_iata[0]]
    if len(unique_iata) >= 2:
        dest_city_val = IATA_CITY[unique_iata[-1]]

    src_match = re.search(
        r'(?i:\b(?:Source|Origin|Departure\s*City|Departing\s*From)\b)[\s:-]*([A-Za-z\s]{2,30})',
        text
    )
    if src_match:
        s_candidate = src_match.group(1).strip().split('\n')[0].strip()
        s_candidate = re.sub(r'(?i)\b(?:Destination|Flight|Departure|Date|PNR|To)\b.*$', '', s_candidate).strip()
        s_upper = s_candidate.upper()
        non_location_words = {"ICAO", "CARBON", "EMISSIONS", "CALCULATOR", "FUNDS", "INCOME", "PAYMENT", "SALARY", "DECLARATION"}
        if s_candidate and not any(w in s_upper for w in non_location_words):
            source_val = s_candidate.title()

    if not source_val:
        route_match = re.search(
            r'(?i:from|origin|departure\s*city)[\s:–-]*([A-Za-z\s]{2,30?})\s*'
            r'(?:to|→|->|destination|arrival\s*city)[\s:–-]*([A-Za-z\s]{2,30})',
            text, re.IGNORECASE
        )
        if route_match:
            source_val = route_match.group(1).strip().title()
            if not dest_city_val:
                dest_city_val = route_match.group(2).strip().title()

    dest_c_match = re.search(
        r'(?i:\b(?:Destination\s*Country|Country\s*of\s*Destination|Country\s*of\s*Travel|Travel\s*Destination)\b)[\s:-]*([A-Za-z\s]{2,30})',
        text
    )
    if dest_c_match:
        dc_candidate = dest_c_match.group(1).strip().split('\n')[0].strip()
        dc_candidate = re.sub(r'(?i)\b(?:Departure|Date|Return|Duration|Flight)\b.*$', '', dc_candidate).strip()
        if dc_candidate:
            dest_country_val = dc_candidate

    if not dest_country_val and dest_city_val:
        dest_country_val = CITY_COUNTRY_MAP.get(dest_city_val.lower(), "")

    if not dest_country_val:
        for kw, country_name in COUNTRY_KEYWORDS.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                src_country = CITY_COUNTRY_MAP.get(source_val.lower(), "India")
                if country_name.lower() != src_country.lower() or kw.lower() in ["united kingdom", "uk"]:
                    dest_country_val = country_name
                    break

    stopwords = {"The", "A", "An", "Date", "Issue", "Time", "Funds", "Payment", "Valid", "Icao Carbon Emissions Calculat"}
    if source_val in stopwords:
        source_val = IATA_CITY.get(unique_iata[0], "") if unique_iata else ""

    result["source"] = source_val
    result["destination_country"] = dest_country_val
    result["origin"] = {"city": source_val, "country": CITY_COUNTRY_MAP.get(source_val.lower(), "")}
    result["destination"] = {"city": dest_city_val, "country": dest_country_val}

    # 5. Departure Date, Return Date & Duration extraction
    MONTHS = r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
    DATE_PAT = (
        r'(?:'
        r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[,.]?\s*'
        r')?'
        r'(?:'
        r'\d{1,2}\s*(?:st|nd|rd|th)?[,\s]+(?:' + MONTHS + r')[a-z]*[,\s]+\d{2,4}'
        r'|\b(?:' + MONTHS + r')[a-z]*[,\s]+\d{1,2}(?!\s*:\s*\d{2})\b[,\s]+\d{2,4}'
        r'|\b(?:' + MONTHS + r')[a-z]*[,\s]+\d{1,2}(?!\s*:\s*\d{2})\b'
        r'|\b\d{1,2}[\-/](?:' + MONTHS + r')[a-z]*[\-/]\d{2,4}\b'
        r'|\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b'
        r'|\b\d{4}[\-/]\d{1,2}[\-/]\d{1,2}\b'
        r'|\b\d{1,2}\s*(?:' + MONTHS + r')[a-z]*\s*\d{2,4}\b'
        r'|\b\d{1,2}\s*(?:' + MONTHS + r')[a-z]*\b'
        r')'
    )

    dep_date_match = re.search(
        r'(?i:\b(?:Departure\s*Date|Departing|Date\s*of\s*Travel|Onward\s*Date|Travel\s*Date|Flight\s*Date)\b)[\s:-]*(' + DATE_PAT + r')',
        text
    )
    if dep_date_match:
        result["departure_date"] = dep_date_match.group(1).strip()

    ret_date_match = re.search(
        r'(?i:\b(?:Return\s*Date|Returning|Inbound\s*Date|Return\s*Flight)\b)[\s:-]*(' + DATE_PAT + r')',
        text
    )
    if ret_date_match:
        result["return_date"] = ret_date_match.group(1).strip()

    all_dates = []
    flight_dates = []
    for dm in re.finditer(DATE_PAT, text, re.IGNORECASE):
        d_str = dm.group(0).strip()
        start_pos = dm.start()
        prefix_text = text[max(0, start_pos - 40):start_pos].lower()
        if any(ign in prefix_text for ign in ["booking", "date of issue", "issue date", "issued:", "printed:", "booking ref"]):
            continue
        if d_str not in all_dates:
            all_dates.append(d_str)
        if any(kw in prefix_text for kw in ["flight", "departure", "departing", "travel", "sat", "sun", "mon", "tue", "wed", "thu", "fri"]):
            if d_str not in flight_dates:
                flight_dates.append(d_str)

    parsed_all_dates = []
    for d_str in all_dates:
        pd = _parse_date(d_str)
        if pd and pd.year >= 2023:
            parsed_all_dates.append((pd, d_str))

    if parsed_all_dates:
        parsed_all_dates.sort(key=lambda x: x[0])
        
        if not result["departure_date"]:
            flight_parsed = [x for x in parsed_all_dates if x[1] in flight_dates]
            if flight_parsed:
                result["departure_date"] = flight_parsed[0][1]
            else:
                result["departure_date"] = parsed_all_dates[0][1]

    has_explicit_return = bool(re.search(r'(?i:\b(?:Return|Inbound|Round\s*Trip|Two\s*Way)\b)', text))
    if not has_explicit_return and len(all_iata) >= 3:
        first_code = all_iata[0]
        if first_code in all_iata[2:]:
            has_explicit_return = True

    if not result["return_date"] and len(parsed_all_dates) >= 2:
        earliest_pd = _parse_date(result["departure_date"])
        latest = parsed_all_dates[-1]
        if earliest_pd and latest[0] > earliest_pd:
            result["return_date"] = latest[1]

    result["onward_date"] = result["departure_date"]

    result["duration"] = _compute_trip_duration(result["departure_date"], result["return_date"])

    return result


def _parse_date(date_str):
    if not date_str:
        return None
    date_str = str(date_str).strip()
    try:
        return datetime.datetime.fromisoformat(date_str[:10])
    except Exception:
        pass
    months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
              'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    
    # Check for DDMMMYY or DDMMMYYYY without delimiters (e.g. 24AUG23, 24AUG)
    m_concat = re.search(r'^(\d{1,2})([a-zA-Z]{3})(\d{2,4})?$', date_str.replace(' ', ''))
    if m_concat:
        day = int(m_concat.group(1))
        month = months.get(m_concat.group(2).lower(), 1)
        year_str = m_concat.group(3)
        if year_str:
            year = int(year_str) if len(year_str) == 4 else int(year_str) + 2000
        else:
            year = datetime.datetime.now().year
        try:
            return datetime.datetime(year, month, day)
        except:
            return None

    match = re.search(r'(\d{1,4})[\-\/\s]+([a-zA-Z]{3,}|\d{1,2})[\-\/\s]+(\d{1,4})', date_str)
    if match:
        p1, p2, p3 = match.groups()
        if len(p1) == 4:
            year, p_other, day = int(p1), p2, int(p3)
        elif len(p3) == 4:
            year, p_other, day = int(p3), p2, int(p1)
        else:
            year, p_other, day = int(p3) + 2000, p2, int(p1)
        month = int(p_other) if p_other.isdigit() else months.get(p_other[:3].lower(), 1)
        try:
            return datetime.datetime(year, month, day)
        except ValueError:
            pass
    return None


def _compute_trip_duration(onward: str, ret: str) -> str:
    d1 = _parse_date(onward)
    d2 = _parse_date(ret)
    if d1 and d2 and d2 > d1:
        days = (d2 - d1).days
        if days == 1:
            return "1 day"
        return f"{days} days"
    return ""


def ticket(pdf_path):
    """Main API for flight ticket reading."""
    return extract_ticket_details(pdf_path)
