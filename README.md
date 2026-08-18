# OpenBharatOCR

A **local, offline OCR library** for Indian government documents.
All processing happens on your machine — no document data is sent to any external service.

---

## Supported documents

| Document | Fields extracted |
|---|---|
| **PAN card** | PAN number, name, father's name, date of birth |
| **Aadhaar (front)** | Name, date of birth, gender, Aadhaar number |
| **Aadhaar (back)** | Relative name, relation type, address, pincode |
| **Passport (front)** | Passport number, name, gender, nationality, DOB, issue date, expiry, place of birth/issue, MRZ |
| **Passport (back)** | Father/guardian name, mother name, spouse name, address, pincode, file number |
| Driving licence | Licence number, name, dates, address, vehicle authorisations |
| Voter ID (front/back) | Voter ID, names, gender, DOB, address |
| Vehicle registration | RC details |
| Water bill | Bill details |
| Birth certificate | Certificate fields |
| Degree certificate | Certificate fields |

---

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR (for passport, driving licence, voter ID)

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### Install the library

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

---

## Model setup

PaddleOCR downloads its models on the **first run** and caches them locally.
After that, the library works fully offline.

```python
# Trigger the one-time download
import openbharatocr
openbharatocr.pan("any_image.jpg")   # models downloaded here
```

To point PaddleOCR at a custom model directory, set the environment variable:

```bash
export PADDLE_HOME=/path/to/your/paddle/cache
```

---

## Usage

### PAN card

```python
import openbharatocr

result = openbharatocr.pan("pan.jpg")
print(result["pan_number"])
print(result["name"])
print(result["father_name"])
print(result["date_of_birth"])
print(result["extraction_confidence"])   # "high" / "medium" / "low"
```

### Aadhaar — front only

```python
result = openbharatocr.aadhaar("aadhaar-front.jpeg")
fields = result["fields"]
print(fields["name"])
print(fields["date_of_birth"])
print(fields["gender"])
print(fields["aadhaar_number"])
```

### Aadhaar — front + back

```python
result = openbharatocr.aadhaar("aadhaar-front.jpeg", "aadhaar-back.jpeg")
fields = result["fields"]
print(fields["address"])
print(fields["pincode"])
```

### Passport — front only

```python
result = openbharatocr.passport("passport-front.jpeg")
print(result["passport_number"])
print(result["name"])
print(result["date_of_birth"])
print(result["expiry_date"])
print(result["mrz"])
```

### Passport — front + back

```python
result = openbharatocr.passport("passport-front.jpeg", "passport-back.jpeg")
print(result["father_name"])
print(result["mother_name"])
print(result["address"])
```

### OCRClient (object-oriented API)

```python
client = openbharatocr.OCRClient()
pan_result      = client.pan("pan.jpg")
aadhaar_result  = client.aadhaar("aadhaar-front.jpeg")
passport_result = client.passport("front.jpeg", "back.jpeg")
```

---

## Output format

### PAN

```json
{
  "pan_number": "ABCDE1234F",
  "name": "Amit Kumar Sharma",
  "father_name": "Rajesh Kumar Sharma",
  "date_of_birth": "15/08/1990",
  "extraction_confidence": "high",
  "confidence_score": 100,
  "confidence_score_type": "completeness",
  "field_confidence": {
    "pan_number": 0.97,
    "name": 0.91,
    "father_name": 0.88,
    "date_of_birth": 0.93
  },
  "raw_text": ["GOVT OF INDIA", "AMIT KUMAR SHARMA", "..."]
}
```

### Aadhaar (front)

```json
{
  "document_type": "aadhaar_front",
  "fields": {
    "name": "Amit Kumar Sharma",
    "date_of_birth": "15/08/1990",
    "gender": "Male",
    "aadhaar_number": "1234 5678 9012"
  },
  "raw_text": "..."
}
```

### Passport

```json
{
  "passport_number": "C7162010",
  "name": "Biswanath Adhikari",
  "surname": "Adhikari",
  "given_names": "Biswanath",
  "gender": "Male",
  "nationality": "IND",
  "date_of_birth": "01-01-1989",
  "date_of_issue": "01-01-2015",
  "expiry_date": "28-01-2035",
  "place_of_birth": "KOLKATA",
  "place_of_issue": "DELHI",
  "father_name": "Ramesh Adhikari",
  "mother_name": "",
  "spouse_name": "",
  "address": {"raw": "...", "lines": [], "pincode": ""},
  "file_number": "",
  "mrz": {"line1": "P<IND...", "line2": "C716...", "valid": true},
  "extraction_confidence": "high",
  "raw_text_front": "...",
  "raw_text_back": "..."
}
```

---

## Developer test script

```bash
python test_ocr.py pan      openbharatocr/test_images/pan_final.jpg
python test_ocr.py aadhaar  openbharatocr/test_images/aadhaar-front.jpeg
python test_ocr.py passport openbharatocr/test_images/pass-front.jpeg
python test_ocr.py passport openbharatocr/test_images/pass-front.jpeg \
                            openbharatocr/test_images/pass-back.jpeg
```

## Running unit tests

```bash
pip install pytest
pytest openbharatocr/unit_tests/ -v
```

---

## Offline / local processing

OpenBharatOCR processes documents entirely on your machine.

- **PAN and Aadhaar**: use PaddleOCR. Models are downloaded once to the local cache on first run; subsequent calls are fully offline.
- **Passport, driving licence, voter ID**: use Tesseract, which is a local binary with no network dependency.
- **No API keys required.**
- **No document data is sent anywhere.**

---

## Debug logging

To enable verbose logging (useful during development):

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Normal usage is silent — no PII is written to stdout or logs.

---

## Limitations

- OCR accuracy depends on image quality. Use clear, well-lit, unobscured images.
- PAN extraction uses spatial heuristics; unusual card layouts may yield lower confidence.
- Aadhaar masking (where the first 8 digits are hidden) is not handled.
- Passport MRZ parsing assumes the standard ICAO TD3 format.
- Hindi/Devanagari fields on PAN cards are partially supported.
- PaddleOCR models are large (~500 MB). Ensure sufficient disk space.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
