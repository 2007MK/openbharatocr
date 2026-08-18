#!/usr/bin/env python3
"""
Developer smoke-test for OpenBharatOCR.

Usage:
    python test_ocr.py pan      openbharatocr/test_images/pan_final.jpg
    python test_ocr.py aadhaar  openbharatocr/test_images/aadhaar-front.jpeg
    python test_ocr.py passport openbharatocr/test_images/pass-front.jpeg
    python test_ocr.py passport openbharatocr/test_images/pass-front.jpeg \
                                openbharatocr/test_images/pass-back.jpeg

All paths are relative to the project root.
Real document images in test_images/ are git-ignored.
"""

import sys
import json
import logging

logging.basicConfig(level=logging.WARNING)

import openbharatocr


def _usage():
    print(__doc__)
    sys.exit(1)


def main():
    args = sys.argv[1:]
    if not args:
        _usage()

    doc_type = args[0].lower()

    if doc_type == "pan":
        if len(args) < 2:
            _usage()
        result = openbharatocr.pan(args[1])

    elif doc_type == "aadhaar":
        if len(args) < 2:
            _usage()
        back = args[2] if len(args) > 2 else None
        result = openbharatocr.aadhaar(args[1], back)

    elif doc_type == "passport":
        if len(args) < 2:
            _usage()
        back = args[2] if len(args) > 2 else None
        result = openbharatocr.passport(args[1], back)

    else:
        print(f"Unknown document type: {doc_type!r}")
        _usage()

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()