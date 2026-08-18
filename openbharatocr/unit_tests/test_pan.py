# from openbharatocr.ocr.pan import (
#     clean_input,
#     extract_all_names,
#     extract_pan,
#     extract_dob,
#     extract_pan_details,
# )
# import pytest
# from unittest import mock, TestCase


# class Test_clean_input(TestCase):
#     def test_clean_input_with_newlines(self):
#         match = ["Amit\nKumar", "Sunita\nKumar\nAnil"]
#         expected_output = ["Amit", "Kumar", "Sunita", "Kumar", "Anil"]
#         assert clean_input(match) == expected_output

#     def test_clean_input_without_newlines(self):
#         match = ["AMIT KUMAR", "SUNITA KUMAR"]
#         expected_output = ["AMIT KUMAR", "SUNITA KUMAR"]
#         assert clean_input(match) == expected_output

#     def test_clean_input_with_empty_string(self):
#         match = [""]
#         expected_output = []
#         assert clean_input(match) == expected_output

#     def test_clean_input_with_none(self):
#         with pytest.raises(TypeError):
#             clean_input(None)


# class Test_extract_all_names(TestCase):
#     def test_extract_all_names(self):
#         input_string = """
#         This is a sample input string.
#         AMIT KUMAR
#         SUNITA KUMAR
#         INCOME TAX DEPARTMENT INDIA
#         ANIL JOSHI
#         GOVT OF INDIA
#         """
#         expected_output = ["AMIT KUMAR", "SUNITA KUMAR", "ANIL JOSHI"]
#         assert extract_all_names(input_string) == expected_output

#     def test_extract_all_names_individual(slef):
#         assert extract_all_names("") == []

#         input_string = "This is a string without any names."
#         assert extract_all_names(input_string) == []

#         input_string = "INDIA GOVT TAX DEPARTMENT"
#         assert extract_all_names(input_string) == []

#         input_string = "ABC\nXYZ"
#         assert extract_all_names(input_string) == []

#     def test_extract_all_names_with_empty_input(self):
#         input_text = ""
#         expected_output = []
#         assert extract_all_names(input_text) == expected_output

#     def test_extract_all_names_with_invalid_input(self):
#         with pytest.raises(TypeError):
#             extract_all_names(None)


# class Test_extract_pan(TestCase):
#     def test_extract_pan_valid_input(self):
#         input_text = "Amit Kumar's PAN number is ABCDE1234F"
#         expected_output = "ABCDE1234F"
#         assert extract_pan(input_text) == expected_output

#     def test_extract_pan_no_match(self):
#         input_text = "This is a random text without a PAN number"
#         expected_output = ""
#         assert extract_pan(input_text) == expected_output

#     def test_extract_pan_invalid_format(self):
#         input_text = "This is an invalid PAN number: ABCDE12345"
#         assert extract_pan(input_text) == ""

#     def test_extract_pan_invalid_input(self):
#         with pytest.raises(TypeError):
#             extract_pan(None)
#         with pytest.raises(TypeError):
#             extract_pan(123)


# class Test_extract_dob(TestCase):
#     def test_extract_dob_valid_formats(self):
#         inputs = [
#             "Amit Kumar's date of birth is 01/01/1990",
#             "15-12-1985 is sunita kumar's DOB",
#             "Rajiv's birth date is 30.06.78",
#         ]
#         expected_outputs = ["01/01/1990", "15-12-1985", "30.06.78"]
#         for input_str, expected_output in zip(inputs, expected_outputs):
#             assert extract_dob(input_str) == expected_output

#     def test_extract_dob_invalid_formats(self):
#         assert extract_dob("Amit Kumar's DOB is 1990/01/01") == ""
#         assert extract_dob("DOB: 01/Jan/1990") == ""
#         assert extract_dob("My DOB is 01:01:1990") == ""

#         assert extract_dob("DOB: 15 03 1985") == ""

#     def test_extract_dob_invalid_input(self):
#         with pytest.raises(TypeError):
#             extract_dob(None)


# class Test_extract_pan_details(TestCase):
#     def test_image_without_pan_details(self):
#         # Mock the image reading and text extraction
#         with mock.patch("PIL.Image.open") as mock_open, mock.patch(
#             "pytesseract.image_to_string"
#         ) as mock_image_to_string, mock.patch("imghdr.what") as mock_what:
#             mock_image = mock.MagicMock()
#             mock_open.return_value = mock_image
#             mock_image_to_string.return_value = "This is a test without PAN details."
#             mock_what.return_value = "jpeg"

#             result = extract_pan_details("dummy_img.jpg")

#             assert result["Full Name"] == ""
#             assert result["Parent's Name"] == ""
#             assert result["Date of Birth"] == ""
#             assert result["PAN Number"] == ""

#             mock_open.assert_called_once_with("dummy_img.jpg")
#             mock_image_to_string.assert_called_once_with(mock_image)
#             mock_what.assert_called_once_with("dummy_img.jpg")

#             with mock.patch(
#                 "openbharatocr.ocr.pan.extract_all_names"
#             ) as mock_extract_all_names, mock.patch(
#                 "openbharatocr.ocr.pan.extract_dob"
#             ) as mock_extract_dob, mock.patch(
#                 "openbharatocr.ocr.pan.extract_pan"
#             ) as mock_extract_pan:
#                 mock_extract_all_names.return_value = []
#                 mock_extract_dob.return_value = ""
#                 mock_extract_pan.return_value = ""

#                 result = extract_pan_details("dummy_img.jpg")

#                 assert result["Full Name"] == ""
#                 assert result["Parent's Name"] == ""
#                 assert result["Date of Birth"] == ""
#                 assert result["PAN Number"] == ""

#                 mock_extract_all_names.assert_called_once_with(
#                     "This is a test without PAN details."
#                 )
#                 mock_extract_dob.assert_called_once_with(
#                     "This is a test without PAN details."
#                 )
#                 mock_extract_pan.assert_called_once_with(
#                     "This is a test without PAN details."
#                 )


########## NEW TEST CODE ###########
from openbharatocr.ocr.pan import PANCardExtractor
import pytest
from unittest import mock, TestCase
import numpy as np
import json


class Test_clean_text(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_clean_text_with_spaces(self):
        text = "  AMIT   KUMAR  "
        expected_output = "AMIT KUMAR"
        assert self.extractor.clean_text(text) == expected_output

    def test_clean_text_with_special_chars(self):
        text = "AMIT@#$%KUMAR"
        expected_output = "AMITKUMAR"
        assert self.extractor.clean_text(text) == expected_output

    def test_clean_text_with_empty_string(self):
        text = ""
        expected_output = ""
        assert self.extractor.clean_text(text) == expected_output

    def test_clean_text_with_numbers_and_letters(self):
        text = "ABCDE1234F"
        expected_output = "ABCDE1234F"
        assert self.extractor.clean_text(text) == expected_output


class Test_clean_name(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_clean_name_normal(self):
        name = "AMIT KUMAR SHARMA"
        expected_output = "Amit Kumar Sharma"
        assert self.extractor.clean_name(name) == expected_output

    def test_clean_name_with_titles(self):
        name = "SHRI AMIT KUMAR"
        expected_output = "Amit Kumar"
        assert self.extractor.clean_name(name) == expected_output

    def test_clean_name_with_multiple_spaces(self):
        name = "  AMIT    KUMAR  "
        expected_output = "Amit Kumar"
        assert self.extractor.clean_name(name) == expected_output

    def test_clean_name_with_empty_string(self):
        name = ""
        expected_output = ""
        assert self.extractor.clean_name(name) == expected_output


class Test_is_valid_name(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_is_valid_name_valid_names(self):
        assert self.extractor.is_valid_name("AMIT KUMAR") == True
        assert self.extractor.is_valid_name("Rajesh Kumar Sharma") == True
        assert self.extractor.is_valid_name("Sunita Devi") == True

    def test_is_valid_name_invalid_names(self):
        assert self.extractor.is_valid_name("GOVT OF INDIA") == False
        assert self.extractor.is_valid_name("INCOME TAX") == False
        assert self.extractor.is_valid_name("123456") == False
        assert self.extractor.is_valid_name("A") == False
        assert self.extractor.is_valid_name("") == False
        assert self.extractor.is_valid_name("PERMANENT ACCOUNT") == False

    def test_is_valid_name_with_numbers(self):
        assert self.extractor.is_valid_name("AMIT123") == False


class Test_find_pan_number(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_find_pan_number_valid_pan(self):
        text_data = [{"text": "ABCDE1234F", "confidence": 0.9}]
        expected_output = "ABCDE1234F"
        assert self.extractor.find_pan_number(text_data) == expected_output

    def test_find_pan_number_no_match(self):
        text_data = [{"text": "AMIT KUMAR", "confidence": 0.9}]
        expected_output = None
        assert self.extractor.find_pan_number(text_data) == expected_output

    def test_find_pan_number_invalid_format(self):
        text_data = [{"text": "ABCD1234F", "confidence": 0.9}]
        assert self.extractor.find_pan_number(text_data) is None

    def test_find_pan_number_with_spaces(self):
        text_data = [{"text": "ABCD E123 4F", "confidence": 0.9}]
        expected_output = "ABCDE1234F"
        assert self.extractor.find_pan_number(text_data) == expected_output


class Test_find_dates(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_find_dates_valid_formats(self):
        text_data = [
            {"text": "Date of Birth: 15/08/1990", "confidence": 0.9},
            {"text": "DOB: 20-12-1985", "confidence": 0.8},
            {"text": "Born: 05.03.1995", "confidence": 0.85},
        ]
        result = self.extractor.find_dates(text_data)
        assert "15/08/1990" in result
        assert "20-12-1985" in result
        assert "05.03.1995" in result

    def test_find_dates_no_dates(self):
        text_data = [
            {"text": "AMIT KUMAR", "confidence": 0.9},
            {"text": "GOVT OF INDIA", "confidence": 0.8},
        ]
        result = self.extractor.find_dates(text_data)
        assert result == []


class Test_validate_pan(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_validate_pan_valid_pans(self):
        assert self.extractor.validate_pan("ABCDE1234F") == True
        assert self.extractor.validate_pan("ZYXWV9876A") == True

    def test_validate_pan_invalid_pans(self):
        assert self.extractor.validate_pan("ABCD1234F") == False   # only 4 letters prefix
        assert self.extractor.validate_pan("ABCDE12345") == False  # ends with digit
        # Lowercase is normalized to upper so "abcde1234f" IS valid — that's correct
        assert self.extractor.validate_pan("abcde1234f") == True
        assert self.extractor.validate_pan("") == False
        assert self.extractor.validate_pan(None) == False


class Test_validate_date(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_validate_date_valid_dates(self):
        assert self.extractor.validate_date("15/08/1990") == True
        assert self.extractor.validate_date("01-01-1980") == True
        assert self.extractor.validate_date("31.12.1995") == True

    def test_validate_date_invalid_dates(self):
        assert self.extractor.validate_date("32/01/1990") == False
        assert self.extractor.validate_date("15/13/1990") == False
        assert self.extractor.validate_date("15/08/1800") == False
        assert self.extractor.validate_date("15/08/2030") == False
        assert self.extractor.validate_date("") == False
        assert self.extractor.validate_date("invalid") == False


class Test_extract_pan_details(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_extract_pan_details_success(self):
        # Mock the preprocessing and the shared engine call
        with mock.patch.object(
            self.extractor, "preprocess_image"
        ) as mock_preprocess, mock.patch(
            "openbharatocr.ocr.pan.extract_text_with_coords_paddle"
        ) as mock_extract_text:

            mock_preprocess.return_value = np.ones((100, 100, 3), dtype=np.uint8)

            mock_text_data = [
                {"text": "GOVT OF INDIA", "confidence": 0.95, "center_y": 20.0, "center_x": 100.0, "bbox": []},
                {"text": "AMIT KUMAR SHARMA", "confidence": 0.89, "center_y": 110.0, "center_x": 100.0, "bbox": []},
                {"text": "RAJESH KUMAR SHARMA", "confidence": 0.87, "center_y": 140.0, "center_x": 100.0, "bbox": []},
                {"text": "ABCDE1234F", "confidence": 0.92, "center_y": 180.0, "center_x": 100.0, "bbox": []},
                {"text": "15/08/1990", "confidence": 0.85, "center_y": 210.0, "center_x": 100.0, "bbox": []},
            ]
            mock_extract_text.return_value = mock_text_data

            result = self.extractor.extract_pan_details("test_image.jpg")

            assert "pan_number" in result
            assert "name" in result
            assert "father_name" in result
            assert "date_of_birth" in result
            assert result["pan_number"] == "ABCDE1234F"
            assert result["date_of_birth"] == "15/08/1990"

    def test_extract_pan_details_preprocessing_error(self):
        with mock.patch.object(self.extractor, "preprocess_image") as mock_preprocess:
            mock_preprocess.side_effect = ValueError("Could not read image")

            result = self.extractor.extract_pan_details("invalid_image.jpg")

            assert "error" in result
            assert "Could not read image" in result["error"]

    def test_extract_pan_details_no_text(self):
        with mock.patch.object(
            self.extractor, "preprocess_image"
        ) as mock_preprocess, mock.patch(
            "openbharatocr.ocr.pan.extract_text_with_coords_paddle"
        ) as mock_extract_text:

            mock_preprocess.return_value = np.ones((100, 100, 3), dtype=np.uint8)
            mock_extract_text.return_value = []

            result = self.extractor.extract_pan_details("test_image.jpg")

            assert result["error"] == "No text could be extracted from the image"

    def test_extract_pan_details_without_pan_details(self):
        with mock.patch.object(
            self.extractor, "preprocess_image"
        ) as mock_preprocess, mock.patch(
            "openbharatocr.ocr.pan.extract_text_with_coords_paddle"
        ) as mock_extract_text:

            mock_preprocess.return_value = np.ones((100, 100, 3), dtype=np.uint8)

            mock_text_data = [
                {"text": "GOVT OF INDIA", "confidence": 0.95, "center_y": 20.0, "center_x": 100.0, "bbox": []},
                {"text": "INCOME TAX DEPARTMENT", "confidence": 0.93, "center_y": 50.0, "center_x": 100.0, "bbox": []},
            ]
            mock_extract_text.return_value = mock_text_data

            result = self.extractor.extract_pan_details("test_image.jpg")

            assert result["pan_number"] is None
            assert result["name"] == ""
            assert result["father_name"] == ""
            assert result["date_of_birth"] is None


class Test_preprocess_image(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    @mock.patch("cv2.imread")
    def test_preprocess_image_success(self, mock_imread):
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mock_imread.return_value = mock_image

        result = self.extractor.preprocess_image("test_image.jpg")

        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_imread.assert_called_once_with("test_image.jpg")

    @mock.patch("cv2.imread")
    def test_preprocess_image_failure(self, mock_imread):
        mock_imread.return_value = None

        with pytest.raises(ValueError, match="Could not read image from"):
            self.extractor.preprocess_image("invalid_image.jpg")


class Test_save_results(TestCase):
    """save_results was removed in the refactoring (wrote files to disk)."""

    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_save_results_not_present(self):
        # save_results was intentionally removed — verify it's gone
        assert not hasattr(self.extractor, "save_results"), (
            "save_results should have been removed in the refactoring"
        )


class Test_extract_names_with_keywords(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def _item(self, text, y, confidence=0.9, x=100.0):
        return {"text": text, "confidence": confidence, "center_y": y, "center_x": x, "bbox": []}

    def test_extract_names_with_keywords_success(self):
        text_data = [
            self._item("Name:", 100),
            self._item("AMIT KUMAR SHARMA", 110),
            self._item("Father's Name:", 130),
            self._item("RAJESH KUMAR", 140),
        ]

        name, father_name = self.extractor.extract_names_with_keywords(text_data)

        assert name is not None
        assert father_name is not None

    def test_extract_names_with_keywords_no_keywords(self):
        text_data = [
            self._item("AMIT KUMAR SHARMA", 110),
            self._item("RAJESH KUMAR", 140),
        ]

        name, father_name = self.extractor.extract_names_with_keywords(text_data)

        assert name is None
        assert father_name is None


class Test_extract_names_positional(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def _item(self, text, y, confidence=0.9):
        return {"text": text, "confidence": confidence, "center_y": y, "center_x": 100.0, "bbox": []}

    def test_extract_names_positional_success(self):
        text_data = [
            self._item("GOVT OF INDIA", 50),
            self._item("AMIT KUMAR SHARMA", 100),
            self._item("RAJESH KUMAR SHARMA", 130),
            self._item("ABCDE1234F", 160),
        ]

        name, father_name = self.extractor.extract_names_positional(text_data)

        # Should extract some names based on position
        assert name is not None or father_name is not None

    def test_extract_names_positional_no_valid_names(self):
        text_data = [
            self._item("GOVT OF INDIA", 50),
            self._item("INCOME TAX", 80),
            self._item("DEPARTMENT", 110),
        ]

        name, father_name = self.extractor.extract_names_positional(text_data)

        assert name is None
        assert father_name is None


class Test_find_names_improved(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def _item(self, text, y, confidence=0.9):
        return {"text": text, "confidence": confidence, "center_y": y, "center_x": 100.0, "bbox": []}

    def test_find_names_improved_success(self):
        text_data = [
            self._item("AMIT KUMAR", 100),
            self._item("RAJESH SHARMA", 130),
        ]

        result = self.extractor.find_names_improved(text_data)

        assert "name" in result
        assert "father_name" in result

    def test_find_names_improved_identical_names(self):
        text_data = [
            self._item("AMIT KUMAR", 100),
            self._item("AMIT KUMAR", 130),
            self._item("RAJESH SHARMA", 160, 0.85),
        ]

        result = self.extractor.find_names_improved(text_data)

        # Should handle identical names and find alternatives
        assert result["name"] != result["father_name"] or not result["father_name"]


# ===========================================================================
# Regression tests: father's-name extraction (COVT OR INDLA false-positive)
#
# Real card scenario:
#   BALWINDER SINGH     ← applicant name
#   Father's Name       ← label
#   TWITTERPREET SINGH  ← true father's name (nearest valid line below label)
#   COVT OR INDLA       ← OCR misread of 'GOVT OF INDIA' header
#   14/05/1995          ← DOB
# ===========================================================================

class TestFatherNameRegression(TestCase):
    """
    Regression tests to ensure 'COVT OR INDLA' (or similar OCR noise from
    'GOVT OF INDIA') is never selected as the father's name when the true
    father's name is present directly below the label.
    """

    def setUp(self):
        self.extractor = PANCardExtractor()

    def _item(self, text, y, x=200.0, confidence=0.92, bbox=None):
        """Build a minimal OCR item dict with bounding box."""
        if bbox is None:
            # Simulate a bounding box: 4 corners, 200 px wide, 30 px tall
            left, right = x - 100, x + 100
            top, bot = y - 15, y + 15
            bbox = [[left, top], [right, top], [right, bot], [left, bot]]
        return {
            "text": text,
            "confidence": confidence,
            "center_y": y,
            "center_x": x,
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    # 1. is_valid_name must reject OCR-noisy govt-header text
    # ------------------------------------------------------------------

    def test_covt_or_indla_rejected_by_is_valid_name(self):
        """'COVT OR INDLA' should fail is_valid_name (contains 'covt')."""
        assert self.extractor.is_valid_name("COVT OR INDLA") is False

    def test_govt_of_india_rejected_by_is_valid_name(self):
        assert self.extractor.is_valid_name("GOVT OF INDIA") is False

    def test_govl_or_india_rejected_by_is_valid_name(self):
        assert self.extractor.is_valid_name("GOVL OR INDIA") is False

    def test_income_tax_department_rejected_by_is_valid_name(self):
        assert self.extractor.is_valid_name("INCOME TAX DEPARTMENT") is False

    # ------------------------------------------------------------------
    # 2. _find_nearest_valid_below_label picks TWITTERPREET, not COVT noise
    # ------------------------------------------------------------------

    def test_nearest_below_picks_twitterpreet_not_covt(self):
        """
        Layout (y increases downward):
          y=120  Father's Name   ← label
          y=155  TWITTERPREET SINGH ← correct value (nearest valid below)
          y=190  COVT OR INDLA   ← OCR noise (further below + rejected by is_valid_name)
        """
        label = self._item("Father's Name", 120)
        data = [
            self._item("BALWINDER SINGH", 80),
            label,
            self._item("TWITTERPREET SINGH", 155),
            self._item("COVT OR INDLA", 190, confidence=0.97),  # higher OCR conf
            self._item("14/05/1995", 230),
        ]

        blocked = [
            "NAME", "FATHER", "FATHERS", "DATE", "BIRTH",
            "SIGNATURE", "PERMANENT", "ACCOUNT", "NUMBER",
            "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA",
        ]
        result = self.extractor._find_nearest_valid_below_label(label, data, blocked)

        assert result is not None, "Expected a candidate below the label"
        assert "TWITTERPREET" in result["text"].upper(), (
            f"Expected TWITTERPREET SINGH, got {result['text']!r}"
        )

    def test_nearest_below_rejects_when_only_noise_below(self):
        """If only noise is below the label, return None rather than garbage."""
        label = self._item("Father's Name", 120)
        data = [
            label,
            self._item("COVT OR INDLA", 155, confidence=0.97),
            self._item("GOVT OF INDIA", 190),
        ]
        blocked = [
            "NAME", "FATHER", "FATHERS", "DATE", "BIRTH",
            "SIGNATURE", "PERMANENT", "ACCOUNT", "NUMBER",
            "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA",
        ]
        result = self.extractor._find_nearest_valid_below_label(label, data, blocked)
        assert result is None

    def test_nearest_below_prefers_closer_item_over_higher_confidence(self):
        """
        When the correct value is closer but has lower OCR confidence than
        a more-distant noise item, proximity should win.
        """
        label = self._item("Father's Name", 100)
        closer = self._item("TWITTERPREET SINGH", 130, confidence=0.75)
        farther_high_conf = self._item("RAJINDER KUMAR", 180, confidence=0.99)
        data = [label, closer, farther_high_conf]
        blocked = [
            "NAME", "FATHER", "FATHERS", "DATE", "BIRTH",
            "SIGNATURE", "PERMANENT", "ACCOUNT", "NUMBER",
            "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA",
        ]
        result = self.extractor._find_nearest_valid_below_label(label, data, blocked)
        assert result is not None
        assert "TWITTERPREET" in result["text"].upper()

    # ------------------------------------------------------------------
    # 3. extract_names_with_keywords integration
    # ------------------------------------------------------------------

    def test_extract_names_with_keywords_real_card_layout(self):
        """
        Simulates the exact OCR token layout from the failing real card:
          - Applicant name: BALWINDER SINGH
          - Father label present
          - Father value: TWITTERPREET SINGH (nearest below label)
          - Noise: COVT OR INDLA (rejected by is_valid_name)
        """
        data = [
            self._item("INCOME TAX DEPARTMENT", 30),
            self._item("BALWINDER SINGH", 90),
            self._item("Father's Name", 130),
            self._item("TWITTERPREET SINGH", 165),
            self._item("COVT OR INDLA", 195, confidence=0.97),
            self._item("14/05/1995", 220),
        ]
        name, father_name = self.extractor.extract_names_with_keywords(data)

        assert father_name is not None, "Father's name should be extracted"
        assert "TWITTERPREET" in father_name.upper(), (
            f"Expected TWITTERPREET SINGH as father name, got {father_name!r}"
        )
        # Applicant name should NOT be the father's name
        if name:
            assert name.upper() != father_name.upper()

    def test_extract_names_with_keywords_covt_not_used(self):
        """COVT OR INDLA must never appear in the father_name field."""
        data = [
            self._item("BALWINDER SINGH", 90),
            self._item("Father's Name", 130),
            self._item("TWITTERPREET SINGH", 165),
            self._item("COVT OR INDLA", 160, confidence=0.99),  # almost same y
        ]
        name, father_name = self.extractor.extract_names_with_keywords(data)
        if father_name:
            assert "COVT" not in father_name.upper()
            assert "INDLA" not in father_name.upper()

    # ------------------------------------------------------------------
    # 4. Full pipeline (mocked engine)
    # ------------------------------------------------------------------

    def test_full_pipeline_real_card_layout(self):
        """
        End-to-end test of extract_pan_details with mocked OCR output
        matching the real failing card's token layout.
        """
        data = [
            self._item("INCOME TAX DEPARTMENT", 30),
            self._item("BWZPS1234R", 60),
            self._item("BALWINDER SINGH", 90),
            self._item("Father's Name", 130),
            self._item("TWITTERPREET SINGH", 165),
            self._item("COVT OR INDLA", 195, confidence=0.97),
            self._item("14/05/1995", 220),
        ]

        with mock.patch.object(
            self.extractor, "preprocess_image"
        ) as mp, mock.patch(
            "openbharatocr.ocr.pan.extract_text_with_coords_paddle"
        ) as me:
            mp.return_value = np.ones((300, 500, 3), dtype=np.uint8)
            me.return_value = data

            result = self.extractor.extract_pan_details("fake.jpg")

        assert result.get("pan_number") == "BWZPS1234R"
        assert result.get("date_of_birth") == "14/05/1995"

        father = result.get("father_name", "")
        assert "TWITTERPREET" in father.upper(), (
            f"Expected TWITTERPREET SINGH as father name, got {father!r}"
        )
        assert "COVT" not in father.upper(), (
            f"'COVT OR INDLA' noise leaked into father_name: {father!r}"
        )
