"""
Voter ID OCR extractor.

Uses Tesseract (pytesseract) for OCR.  Temporary files are handled
in-memory where possible.
"""

import logging
import os
import re
import tempfile
import uuid

import cv2
import numpy as np
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

YOLO_CFG = os.environ.get("YOLO_CFG", "yolov3_custom.cfg")
YOLO_WEIGHT = os.environ.get("YOLO_WEIGHT", "yolov3_custom_6000.weights")


def _extract_text(image) -> str:
    return pytesseract.image_to_string(image, config="--psm 6").strip()


def _preprocess_for_bold_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    enhanced = cv2.addWeighted(blurred, 2.5, blurred, -1.0, 0)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sharpened = cv2.filter2D(
        binary, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    )
    return sharpened


def _extract_voter_id(text: str) -> str:
    match = re.search(r"([A-Z]{2,4}[0-9]{6,8})", text)
    return match.group(1).strip() if match else ""


def _extract_names(text: str) -> list:
    candidates = re.findall(r"\b[A-Z][a-z]+\b", text)
    blacklist = {
        "The", "And", "For", "In", "On", "At", "Of", "By",
        "With", "A", "An", "This", "That", "These", "Those",
    }
    return [w for w in candidates if w not in blacklist]


def _extract_uppercase_words(text: str) -> list:
    lines = []
    for line in text.split("\n"):
        words = re.findall(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*", line)
        lines.extend(words)
    return lines


def _extract_gender(text: str) -> str:
    if "female" in text.lower():
        return "Female"
    if "male" in text.lower():
        return "Male"
    return ""


def _extract_date(text: str) -> str:
    match = re.search(r"\b\d{2}[-/.]\d{2}[-/.]\d{2,4}\b", text)
    return match.group(0) if match else ""


def _extract_address(text: str) -> str:
    match = re.search(
        r"(?:Address\s*[:\-]?\s*)?([A-Za-z0-9,.\-\/\s\n]+?\d{6})",
        text, re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _extract_voter_details_yolo(image_path: str) -> dict:
    image = Image.open(image_path)
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHT)
    classes = ["elector", "relation", "voterid"]
    rgb = image.convert("RGB")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, f"{uuid.uuid4()}.jpg")
        rgb.save(tmp_path)
        img = cv2.imread(tmp_path)
        height, width = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

        boxes, confidences, class_ids = [], [], []
        results: dict = {}

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > 0.5:
                    cx, cy, w, h = (
                        detection[:4] * [width, height, width, height]
                    ).astype(int)
                    x, y = cx - w // 2, cy - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            roi = img[y: y + h, x: x + w]
            if roi.size == 0:
                continue
            results[classes[class_ids[i]]] = _extract_text(roi)

    return results


def _extract_voterid_front(image_path: str) -> dict:
    image = Image.open(image_path)
    text = _extract_text(image)
    voter_id = _extract_voter_id(text)
    names = _extract_names(text)
    gender = _extract_gender(text)
    dob = _extract_date(text)

    if not voter_id or not names or not dob:
        # Fallback: preprocess in memory
        raw_img = cv2.imread(image_path)
        if raw_img is not None:
            processed = _preprocess_for_bold_text(raw_img)
            pil_processed = Image.fromarray(processed)
            text2 = _extract_text(pil_processed)
            if not voter_id:
                voter_id = _extract_voter_id(text2)
            if not names:
                candidates = _extract_uppercase_words(text2)
                if len(candidates) >= 2:
                    names = [candidates[-2], candidates[-1]]
            if not dob:
                dob = _extract_date(text2)
            if not gender:
                gender = _extract_gender(text2)

    return {
        "Voter ID": voter_id,
        "Elector's Name": names[0] if len(names) > 0 else "",
        "Father's Name": names[1] if len(names) > 1 else "",
        "Gender": gender,
        "Date of Birth": dob,
    }


def _extract_voterid_back(image_path: str) -> dict:
    text = _extract_text(Image.open(image_path))
    return {
        "Address": _extract_address(text),
        "Date of Issue": _extract_date(text),
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def voter_id_front(image_path: str) -> dict:
    """Extract front-side voter ID information."""
    text = _extract_text(Image.open(image_path))
    if any(kw in text.lower() for kw in ["date", "age", "gender", "sex"]):
        return _extract_voterid_front(image_path)
    return _extract_voter_details_yolo(image_path)


def voter_id_back(image_path: str) -> dict:
    """Extract back-side voter ID information."""
    return _extract_voterid_back(image_path)


# Backward-compatibility / unit test aliases
extract_voter_id = _extract_voter_id
extract_names = _extract_names
extract_gender = _extract_gender
extract_date = _extract_date
extract_address = _extract_address

