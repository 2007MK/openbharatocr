"""
Shared OCR engine for OpenBharatOCR.

Uses RapidOCR (ONNX runtime) as the primary backend — it is 7-20x faster
than PaddleOCR on CPU-only hardware.  PaddleOCR is kept as a fallback.

Both engines are initialized once and reused for all subsequent calls
in the same process (lazy singleton pattern, thread-safe).
"""

import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_rapidocr_instance = None
_paddle_ocr_instance = None


# ------------------------------------------------------------------
# RapidOCR (primary — fast ONNX-based engine)
# ------------------------------------------------------------------

def get_rapid_ocr():
    """
    Return a shared, lazily-initialized RapidOCR instance.

    Thread-safe. Model is loaded the first time this is called and
    reused for all subsequent calls in the same process.
    """
    global _rapidocr_instance

    if _rapidocr_instance is not None:
        return _rapidocr_instance

    with _lock:
        if _rapidocr_instance is not None:
            return _rapidocr_instance

        from rapidocr_onnxruntime import RapidOCR
        logger.info("Initializing RapidOCR (ONNX runtime)...")
        _rapidocr_instance = RapidOCR()
        logger.info("RapidOCR initialized successfully.")
        return _rapidocr_instance


def _run_rapidocr(image):
    """Run RapidOCR inference. Returns raw result list or None."""
    try:
        ocr = get_rapid_ocr()
        result, _ = ocr(image)
        return result  # list of [bbox, text, confidence] or None
    except Exception as exc:
        logger.warning("RapidOCR inference failed: %s", exc)
        return None


# ------------------------------------------------------------------
# PaddleOCR (fallback)
# ------------------------------------------------------------------

def get_paddle_ocr():
    """
    Return a shared, lazily-initialized PaddleOCR instance (fallback).

    Thread-safe. The model is loaded the first time this function is called
    and reused for all subsequent calls in the same process.
    """
    global _paddle_ocr_instance

    if _paddle_ocr_instance is not None:
        return _paddle_ocr_instance

    with _lock:
        if _paddle_ocr_instance is not None:
            return _paddle_ocr_instance

        from paddleocr import PaddleOCR  # imported here to keep startup fast

        _initialization_attempts = [
            # Fast: disable UVDoc (document unwarping) and doc orientation classify
            # These add ~15s/image with no benefit for flat passport scans.
            # Keep textline_orientation for reading angled/rotated text.
            {
                "lang": "en",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
                "enable_mkldnn": False,
            },
            # Fallback without textline orientation
            {
                "lang": "en",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "enable_mkldnn": False,
            },
            # Fallback: original configs
            {"lang": "en", "use_textline_orientation": True, "enable_mkldnn": False},
            {"lang": "en", "use_angle_cls": True, "enable_mkldnn": False},
            {"lang": "en", "enable_mkldnn": False},
            {"enable_mkldnn": False},
        ]

        last_error = None
        for params in _initialization_attempts:
            try:
                logger.info("Initializing PaddleOCR with params: %s", params)
                _paddle_ocr_instance = PaddleOCR(**params)
                logger.info("PaddleOCR initialized successfully.")
                return _paddle_ocr_instance
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(
            "Failed to initialize PaddleOCR with any configuration. "
            f"Last error: {last_error}"
        )


def _run_paddleocr(image):
    """Run PaddleOCR inference with lock. Returns raw result or None."""
    with _lock:
        ocr = get_paddle_ocr()
        try:
            if hasattr(ocr, "predict"):
                return list(ocr.predict(image))
            else:
                return ocr.ocr(image)
        except Exception:
            try:
                return ocr.ocr(image)
            except Exception:
                return None


def _rapidocr_to_text(rapid_result, confidence_threshold: float = 0.3, band_threshold: int = 12) -> str:
    """
    Convert raw RapidOCR output to a clean string preserving reading order.

    Algorithm:
    1. Filter by confidence.
    2. Sort items top-to-bottom (center_y).
    3. Group items into horizontal bands: items within `band_threshold` pixels
       of the current band's average Y are on the same visual line.
    4. Within each band, sort left-to-right (center_x) and join with a space.
    5. Join bands with newlines.

    This prevents the 'WORD1WORD2' concatenation that occurs when adjacent
    text boxes on the same line get separate newlines rather than a space.
    """
    items = []
    for item in rapid_result:
        try:
            bbox = item[0]
            text = str(item[1]).strip()
            conf = float(item[2])
            if conf < confidence_threshold or not text:
                continue
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            cy = sum(ys) / len(ys)
            cx = sum(xs) / len(xs)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y
            actual_height = min(width, height)
            if actual_height == 0:
                actual_height = max(width, height) or 10
            items.append((cy, cx, text, min_x, max_x, actual_height))
        except Exception:
            continue

    if not items:
        return ""

    # Sort by Y first
    items.sort(key=lambda x: x[0])

    # Group into horizontal bands
    bands = []
    current_band = [items[0]]
    current_y = items[0][0]

    for item in items[1:]:
        if abs(item[0] - current_y) <= band_threshold:
            current_band.append(item)
            current_y = sum(i[0] for i in current_band) / len(current_band)
        else:
            bands.append(current_band)
            current_band = [item]
            current_y = item[0]
    bands.append(current_band)

    # Build text: sort each band left-to-right, join with space or newline
    lines = []
    for band in bands:
        band.sort(key=lambda x: x[1])  # sort by center_x
        band_str = ""
        for i, item in enumerate(band):
            if i == 0:
                band_str += item[2]
            else:
                prev = band[i-1]
                dist = item[3] - prev[4]  # min_x of current - max_x of prev
                avg_height = (item[5] + prev[5]) / 2.0
                # If the horizontal gap is larger than 1.5x the font height, it's a separate column
                if dist > avg_height * 1.5:
                    band_str += "\n" + item[2]
                elif dist > avg_height * 0.5:
                    band_str += " | " + item[2]
                else:
                    band_str += " " + item[2]
        lines.append(band_str)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Public extraction functions (same interface as before)
# ------------------------------------------------------------------

def extract_text_paddle(image, confidence_threshold: float = 0.5) -> str:
    """
    Run OCR on *image* (numpy array or file path) and return a
    newline-joined string of all text lines that exceed *confidence_threshold*.

    Tries RapidOCR first (fast ONNX); falls back to PaddleOCR.
    """
    # --- Try RapidOCR first ---
    rapid_result = _run_rapidocr(image)
    if rapid_result is not None:
        texts = _rapidocr_to_text(rapid_result, confidence_threshold)
        if texts:
            return texts

    # --- Fallback: PaddleOCR ---
    raw_result = _run_paddleocr(image)
    if not raw_result:
        return ""

    texts = []
    try:
        for item in raw_result:
            # PaddleX dict format
            if isinstance(item, dict) and "rec_texts" in item:
                for text, score in zip(item.get("rec_texts", []), item.get("rec_scores", [])):
                    if score >= confidence_threshold:
                        texts.append(str(text).strip())

            # Standard list format: [[bbox, [text, conf]], ...]
            elif isinstance(item, list):
                for entry in item:
                    if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
                        continue
                    text_info = entry[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text, conf = str(text_info[0]), float(text_info[1])
                    elif isinstance(text_info, str):
                        text, conf = text_info, 1.0
                    else:
                        continue
                    if conf >= confidence_threshold:
                        texts.append(text.strip())

            elif isinstance(item, str):
                texts.append(item.strip())

    except Exception:
        pass

    return "\n".join(t for t in texts if t)


def extract_text_with_coords_paddle(image, confidence_threshold: float = 0.3) -> list:
    """
    Run OCR and return structured per-token results with bounding boxes.

    Each item in the returned list is a dict:
        {
            "text": str,
            "confidence": float,
            "bbox": list,
            "center_x": float,
            "center_y": float,
        }

    Tries RapidOCR first (fast ONNX); falls back to PaddleOCR.
    """
    # --- Try RapidOCR first ---
    rapid_result = _run_rapidocr(image)
    if rapid_result is not None:
        extracted = []
        for item in rapid_result:
            try:
                bbox = item[0]   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = str(item[1]).strip()
                conf = float(item[2])
                if conf < confidence_threshold or not text:
                    continue
                pts = [[float(p[0]), float(p[1])] for p in bbox]
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                extracted.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": pts,
                    "center_x": cx,
                    "center_y": cy,
                })
            except Exception:
                continue
        if extracted:
            return extracted

    # --- Fallback: PaddleOCR ---
    import numpy as np

    raw_result = _run_paddleocr(image)
    if not raw_result:
        return []

    extracted = []
    try:
        for result_item in raw_result:
            # PaddleX dict format
            if isinstance(result_item, dict) and "rec_texts" in result_item:
                texts = result_item.get("rec_texts", [])
                scores = result_item.get("rec_scores", [])
                polys = result_item.get("rec_polys", [])

                for i, text in enumerate(texts):
                    conf = scores[i] if i < len(scores) else 0.8
                    if conf <= confidence_threshold:
                        continue
                    bbox = polys[i] if i < len(polys) else []
                    if len(bbox) > 0:
                        try:
                            bbox_arr = np.array(bbox)
                            if bbox_arr.ndim == 2:
                                cx = float(np.mean(bbox_arr[:, 0]))
                                cy = float(np.mean(bbox_arr[:, 1]))
                            else:
                                cx = cy = 0.0
                        except Exception:
                            cx = cy = 0.0
                    else:
                        cx = cy = 0.0
                    extracted.append({
                        "text": str(text).strip(),
                        "confidence": float(conf),
                        "bbox": bbox if isinstance(bbox, list) else (bbox.tolist() if hasattr(bbox, "tolist") else bbox),
                        "center_x": cx,
                        "center_y": cy,
                    })

            # Standard list format
            elif isinstance(result_item, list):
                for line in result_item:
                    try:
                        if not (isinstance(line, (list, tuple)) and len(line) >= 2):
                            continue
                        bbox = line[0]
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text, conf = str(text_info[0]), float(text_info[1])
                        else:
                            text, conf = str(text_info), 0.8
                        if conf <= confidence_threshold:
                            continue
                        cx = sum(p[0] for p in bbox) / len(bbox)
                        cy = sum(p[1] for p in bbox) / len(bbox)
                        extracted.append({
                            "text": text.strip(),
                            "confidence": conf,
                            "bbox": bbox,
                            "center_x": cx,
                            "center_y": cy,
                        })
                    except Exception:
                        continue

    except Exception as e:
        import traceback
        import sys
        traceback.print_exc(file=sys.stderr)

    return extracted
