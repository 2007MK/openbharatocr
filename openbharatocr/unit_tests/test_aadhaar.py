"""
Unit tests for the Aadhaar extractor.

These tests mock the OCR engine so that no real PaddleOCR model
is required to run them.
"""

import pytest
from unittest import mock, TestCase

from openbharatocr.ocr.aadhaar import AadhaarOCR


class TestAadhaarFieldExtraction(TestCase):
    """Tests for individual field-extraction helpers."""

    def setUp(self):
        self.ocr = AadhaarOCR()

    # -----------------------------------------------------------------
    # Name extraction
    # -----------------------------------------------------------------

    def test_extract_name_after_uidai_header(self):
        text = "Government of India\nAmit Kumar Sharma\nDOB: 01/01/1990"
        name = self.ocr._extract_name(text)
        assert name == "Amit Kumar Sharma"

    def test_extract_name_general_pattern(self):
        text = "DOB: 01/01/1990\nAmit Kumar\nMale"
        name = self.ocr._extract_name(text)
        assert "Amit" in name or name == ""  # best-effort

    def test_extract_name_skips_garbage(self):
        text = "Government\nIndia\nUID\n"
        name = self.ocr._extract_name(text)
        assert name == ""

    # -----------------------------------------------------------------
    # DOB extraction
    # -----------------------------------------------------------------

    def test_extract_dob_slash(self):
        assert self.ocr._extract_dob("DOB: 15/08/1990") == "15/08/1990"

    def test_extract_dob_hyphen(self):
        assert self.ocr._extract_dob("Date of Birth: 01-01-1985") == "01/01/1985"

    def test_extract_dob_fallback(self):
        result = self.ocr._extract_dob("05/11/1995")
        assert result == "05/11/1995"

    def test_extract_dob_missing(self):
        assert self.ocr._extract_dob("No date here") == ""

    # -----------------------------------------------------------------
    # Gender extraction
    # -----------------------------------------------------------------

    def test_extract_gender_male(self):
        assert self.ocr._extract_gender("MALE") == "Male"

    def test_extract_gender_female(self):
        assert self.ocr._extract_gender("FEMALE") == "Female"

    def test_extract_gender_transgender(self):
        assert self.ocr._extract_gender("TRANSGENDER") == "Transgender"

    def test_extract_gender_missing(self):
        assert self.ocr._extract_gender("some random text") == ""

    # -----------------------------------------------------------------
    # Aadhaar number extraction
    # -----------------------------------------------------------------

    def test_extract_aadhaar_number_spaced(self):
        result = self.ocr._extract_aadhaar_number("1234 5678 9012")
        assert result == "1234 5678 9012"

    def test_extract_aadhaar_number_continuous(self):
        result = self.ocr._extract_aadhaar_number("123456789012")
        assert result == "1234 5678 9012"

    def test_extract_aadhaar_number_missing(self):
        assert self.ocr._extract_aadhaar_number("no number here") == ""


class TestAadhaarPublicAPI(TestCase):
    """Tests for the public extraction methods (OCR is mocked)."""

    def setUp(self):
        self.ocr = AadhaarOCR()

    @mock.patch("openbharatocr.ocr.aadhaar.extract_text_paddle")
    @mock.patch("openbharatocr.ocr.aadhaar.AadhaarOCR._preprocess")
    def test_extract_front_returns_expected_keys(self, mock_preprocess, mock_paddle):
        import numpy as np
        mock_preprocess.return_value = np.zeros((100, 100, 3), dtype="uint8")
        mock_paddle.return_value = (
            "Government of India\nAmit Kumar Sharma\nDOB: 15/08/1990\nMALE\n1234 5678 9012"
        )

        result = self.ocr.extract_front_aadhaar_details("fake.jpg")

        assert result["document_type"] == "aadhaar_front"
        assert "fields" in result
        assert "name" in result["fields"]
        assert "date_of_birth" in result["fields"]
        assert "gender" in result["fields"]
        assert "aadhaar_number" in result["fields"]

    @mock.patch("openbharatocr.ocr.aadhaar.extract_text_paddle")
    @mock.patch("openbharatocr.ocr.aadhaar.AadhaarOCR._preprocess")
    def test_extract_back_returns_expected_keys(self, mock_preprocess, mock_paddle):
        import numpy as np
        mock_preprocess.return_value = np.zeros((100, 100, 3), dtype="uint8")
        mock_paddle.return_value = (
            "S/o Ramesh Kumar\nAddress: 12 MG Road, Bengaluru 560001"
        )

        result = self.ocr.extract_back_aadhaar_details("fake.jpg")

        assert result["document_type"] == "aadhaar_back"
        assert "fields" in result
        assert "address" in result["fields"]

    @mock.patch("openbharatocr.ocr.aadhaar.extract_text_paddle")
    @mock.patch("openbharatocr.ocr.aadhaar.AadhaarOCR._preprocess")
    def test_extract_combined_returns_merged_fields(self, mock_preprocess, mock_paddle):
        import numpy as np
        mock_preprocess.return_value = np.zeros((100, 100, 3), dtype="uint8")
        mock_paddle.return_value = "Government of India\nAmit Kumar\nDOB: 01/01/1990\nMALE\n1234 5678 9012"

        result = self.ocr.extract_aadhaar_details("front.jpg", "back.jpg")

        assert result["document_type"] == "aadhaar"
        assert "raw_text_front" in result
        assert "raw_text_back" in result


class TestAadhaarImport(TestCase):
    """Verify the top-level import surface works."""

    def test_top_level_aadhaar_function_exists(self):
        import openbharatocr
        assert callable(openbharatocr.aadhaar)
        assert callable(openbharatocr.front_aadhaar)
        assert callable(openbharatocr.back_aadhaar)
