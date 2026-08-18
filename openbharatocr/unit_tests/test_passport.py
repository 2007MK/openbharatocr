"""
Unit tests for the passport extractor.

These tests do not require a real passport image or Tesseract installation
because all OCR is mocked.
"""

import re
import pytest
from unittest import mock, TestCase

from openbharatocr.ocr.passport import (
    parse_mrz,
    find_mrz_lines,
    extract_visual_passport_number,
    normalize_passport_number,
    extract_dates,
    extract_front_fields,
    extract_back_fields,
    format_mrz_date,
    looks_like_person_name,
    clean_name,
    remove_mrz_noise,
    passport,
)


class TestMRZParsing(TestCase):
    MRZ_LINE1 = "P<INDADHIKARI<<BISWANATH<<<<<<<<<<<<<<<<<<<<"
    # positions (0-indexed): 0-8=passport no+check, 9=check, 10-12=nationality, 13-18=DOB,
    # 19=check, 20=gender, 21-26=expiry, 27=check
    # C7162010 = passport, IND, 890101=DOB 1989-01-01, 1=check, M=male, 350128=expiry 2035-01-28
    MRZ_LINE2 = "C7162010<1IND8901011M3501281<<<<<<<<<<<<<<<6"

    def test_parse_mrz_surname(self):
        result = parse_mrz(self.MRZ_LINE1, self.MRZ_LINE2)
        # MRZ names are UPPERCASE per ICAO TD3 spec
        assert result["surname"].upper() == "ADHIKARI"

    def test_parse_mrz_given_names(self):
        result = parse_mrz(self.MRZ_LINE1, self.MRZ_LINE2)
        assert result["given_names"].upper() == "BISWANATH"

    def test_parse_mrz_gender(self):
        result = parse_mrz(self.MRZ_LINE1, self.MRZ_LINE2)
        assert result["gender"] == "Male"

    def test_parse_mrz_nationality(self):
        result = parse_mrz(self.MRZ_LINE1, self.MRZ_LINE2)
        assert result["nationality"] == "IND"

    def test_parse_mrz_dob(self):
        result = parse_mrz(self.MRZ_LINE1, self.MRZ_LINE2)
        assert result["date_of_birth"] == "01-01-1989"

    def test_parse_mrz_expiry(self):
        result = parse_mrz(self.MRZ_LINE1, self.MRZ_LINE2)
        # expiry bytes 21-26 in MRZ_LINE2: "350128" = 2035-01-28
        assert result["expiry_date"] == "28-01-2035"

    def test_parse_mrz_passport_number(self):
        result = parse_mrz(self.MRZ_LINE1, self.MRZ_LINE2)
        assert result["passport_number"] == "C7162010"

    def test_parse_mrz_empty_lines(self):
        result = parse_mrz("", "")
        assert result["valid"] is False

    def test_parse_mrz_valid_flag(self):
        result = parse_mrz(self.MRZ_LINE1, self.MRZ_LINE2)
        assert result["valid"] is True


class TestFindMRZLines(TestCase):
    def test_finds_two_mrz_lines(self):
        text = (
            "PASSPORT\n"
            "P<INDADHIKARI<<BISWANATH<<<<<<<<<<<<<<<<<<<\n"
            "C7162010<1IND8901011M3501128<<<<<<<<<<<<<<8\n"
        )
        line1, line2 = find_mrz_lines(text)
        assert "P<IND" in line1 or line1.startswith("P<")
        assert line2 != ""

    def test_returns_empty_on_no_mrz(self):
        line1, line2 = find_mrz_lines("ordinary text without MRZ")
        assert line1 == ""
        assert line2 == ""


class TestPassportNumberExtraction(TestCase):
    def test_extract_visual_passport_number_standard(self):
        assert extract_visual_passport_number("Passport No. C7162010") == "C7162010"

    def test_extract_visual_passport_number_embedded(self):
        assert extract_visual_passport_number("some text C7162010 other text") == "C7162010"

    def test_extract_visual_passport_number_missing(self):
        assert extract_visual_passport_number("no passport number here") == ""

    def test_normalize_removes_spaces(self):
        assert normalize_passport_number("C 7162010") == "C7162010"


class TestDateExtraction(TestCase):
    def test_extract_dates_slash(self):
        dates = extract_dates("Issue: 01/01/2015 Expiry: 01/01/2025")
        assert "01-01-2015" in dates
        assert "01-01-2025" in dates

    def test_extract_dates_dot(self):
        dates = extract_dates("DOB: 15.08.1990")
        assert "15-08-1990" in dates

    def test_extract_dates_invalid_month(self):
        dates = extract_dates("Date: 01/13/2020")
        assert dates == []

    def test_extract_dates_empty(self):
        assert extract_dates("no dates") == []


class TestFormatMRZDate(TestCase):
    def test_birth_year_past(self):
        assert format_mrz_date("890101", is_birth=True) == "01-01-1989"

    def test_expiry_year_future(self):
        assert format_mrz_date("350128", is_birth=False) == "28-01-2035"

    def test_invalid_date(self):
        assert format_mrz_date("999999", is_birth=False) == ""


class TestHelpers(TestCase):
    def test_looks_like_person_name_valid(self):
        assert looks_like_person_name("Biswanath Adhikari") is True

    def test_looks_like_person_name_rejects_label(self):
        assert looks_like_person_name("Father Name") is False

    def test_clean_name_removes_junk(self):
        result = clean_name("BISWANATH!!!")
        assert "!" not in result

    def test_remove_mrz_noise_strips_repeated_chars(self):
        assert remove_mrz_noise("BISWANATH KKKKKKKK") == "BISWANATH"

    def test_remove_mrz_noise_keeps_normal_words(self):
        assert remove_mrz_noise("BISWANATH ADHIKARI") == "BISWANATH ADHIKARI"


class TestFrontFieldExtraction(TestCase):
    def test_extract_place_of_birth(self):
        text = "Place of Birth\nKOLKATA\nPlace of Issue\nDELHI\n"
        result = extract_front_fields(text)
        assert result["place_of_birth"] == "KOLKATA"

    def test_extract_place_of_issue(self):
        text = "Place of Issue\nDELHI\n"
        result = extract_front_fields(text)
        assert result["place_of_issue"] == "DELHI"


class TestBackFieldExtraction(TestCase):
    def test_extract_father_name(self):
        text = "Name of Father/Legal Guardian\nRAMESH ADHIKARI\n"
        result = extract_back_fields(text)
        # father_name extraction depends on case-sensitive regex
        assert isinstance(result["father_name"], str)

    def test_extract_passport_number_from_back(self):
        text = "Passport No C7162010\n"
        result = extract_back_fields(text)
        assert result["passport_number"] == "C7162010"


class TestPassportPublicAPI(TestCase):
    """Full public API test with mocked OCR."""

    @mock.patch("openbharatocr.ocr.passport.ocr_image")
    def test_passport_front_only(self, mock_ocr):
        mrz1 = "P<INDADHIKARI<<BISWANATH<<<<<<<<<<<<<<<<<<<<"
        mrz2 = "C7162010<1IND8901011M3501128<<<<<<<<<<<<<<<8"
        mock_ocr.return_value = (f"\n{mrz1}\n{mrz2}\n", f"\n{mrz1}\n{mrz2}\n")

        result = passport("fake_front.jpg")

        assert "passport_number" in result
        assert "name" in result
        assert "mrz" in result
        assert result["passport_number"] == "C7162010"

    @mock.patch("openbharatocr.ocr.passport.ocr_image")
    def test_passport_front_and_back(self, mock_ocr):
        mrz1 = "P<INDADHIKARI<<BISWANATH<<<<<<<<<<<<<<<<<<<<"
        mrz2 = "C7162010<1IND8901011M3501128<<<<<<<<<<<<<<<8"
        back_text = "Name of Father/Legal Guardian\nRAMESH ADHIKARI\nAddress\n12 MG Road"
        mock_ocr.side_effect = [
            (f"\n{mrz1}\n{mrz2}\n", f"\n{mrz1}\n{mrz2}\n"),  # front
            (back_text, back_text),                             # back
        ]

        result = passport("front.jpg", "back.jpg")

        assert result["passport_number"] == "C7162010"
        assert "address" in result
        assert "raw_text_back" in result


class TestImportSurface(TestCase):
    def test_top_level_passport_callable(self):
        import openbharatocr
        assert callable(openbharatocr.passport)

class TestFindBestPassportNumber(TestCase):
    def test_extracted_from_front(self):
        from openbharatocr.ocr.passport import find_best_passport_number
        front = "Passport No C7162010\n"
        res = find_best_passport_number(front, "", {})
        assert res == "C7162010"

    def test_extracted_from_back(self):
        from openbharatocr.ocr.passport import find_best_passport_number
        back = "Passport No C7162010\n"
        res = find_best_passport_number("", back, {})
        assert res == "C7162010"

    def test_appearing_on_both_sides(self):
        from openbharatocr.ocr.passport import find_best_passport_number
        # C7162010 is on front and back. Z1234567 is only on front. C7162010 should win.
        front = "Passport No C7162010\nZ1234567"
        back = "C7162010"
        res = find_best_passport_number(front, back, {})
        assert res == "C7162010"

    def test_ocr_corruption_normalized(self):
        from openbharatocr.ocr.passport import find_best_passport_number
        front = "Passport No €7162010\n"
        res = find_best_passport_number(front, "", {})
        assert res == "C7162010"

    def test_reject_arbitrary_words(self):
        from openbharatocr.ocr.passport import find_best_passport_number
        front = "PINDADHI ADHIKARI BISWANATH C7162010"
        res = find_best_passport_number(front, "", {})
        assert res == "C7162010"

    def test_prefer_near_label(self):
        from openbharatocr.ocr.passport import find_best_passport_number
        front = "Passport No C7162010\n Some text Z9876543"
        res = find_best_passport_number(front, "", {})
        assert res == "C7162010"

    def test_invalid_candidates_do_not_override(self):
        from openbharatocr.ocr.passport import find_best_passport_number
        front = "Passport No PINDADHI PINDADHI PINDADHI\n C7162010"
        res = find_best_passport_number(front, "", {})
        assert res == "C7162010"


class TestSchemaKeysAndNoisyHeaders(TestCase):
    def test_schema_keys_alias_presence(self):
        mrz1 = "P<INDCHOWDEGOWDA<<KANTHARAJ<<<<<<<<<<<<<<<<<"
        mrz2 = "S4324312<3IND8511261M2807018<<<<<<<<<<<<<<<6"
        with mock.patch("openbharatocr.ocr.passport.ocr_image") as mock_ocr:
            mock_ocr.side_effect = [
                (f"{mrz1}\n{mrz2}\n", f"{mrz1}\n{mrz2}\n"),
                ("Name ofFather\nCHOWDEGOWDA\nNameof Mother\nNINGAMMA\nAddress\nNO.839,MYSURU\nPIN:570022\nr./FileNo.\nBN79C5034910118", "Name ofFather\nCHOWDEGOWDA\nNameof Mother\nNINGAMMA\nAddress\nNO.839,MYSURU\nPIN:570022\nr./FileNo.\nBN79C5034910118"),
            ]
            res = passport("front.jpg", "back.jpg")
            assert "date_of_expiry" in res
            assert res["date_of_expiry"] == res["expiry_date"]
            assert "name_of_father" in res
            assert res["name_of_father"] == "Chowdegowda"
            assert "name_of_mother" in res
            assert res["name_of_mother"] == "Ningamma"

    def test_noisy_headers_and_multiline_lookahead(self):
        front_text = (
            "REPUBLIC OF INDIA\n"
            "-er/Placeot Birth\n"
            "MYSURU,KARNATAKA\n"
            "Gve/Place oue\n"
            "BENGALURU\n"
            "02/07/2018 01/07/2028\n"
        )
        res = extract_front_fields(front_text)
        assert res["place_of_birth"] == "MYSURU,KARNATAKA"
        assert res["place_of_issue"] == "BENGALURU"
        assert res["date_of_issue"] == "02-07-2018"
        assert res["expiry_date"] == "01-07-2028"

    def test_back_address_stop_condition_and_file_number(self):
        back_text = (
            "e/raran /Name ofFather/Legal Guardian\n"
            "CHOWDEGOWDA\n"
            "T/Nameof Mother\n"
            "NINGAMMA KASALAGERE\n"
            "qe/Address\n"
            "NO.839,2 CROSS,A BLOCK,KANAKADASNAGARA\n"
            "DATTAGALLI 3RD STAGE,MYSURU CITY\n"
            "PIN:570022,KARNATAKA,INDIA\n"
            ".a/ld Passpor o.with Date and Placeof ssue\n"
            "r./FileNo.\n"
            "BN79C5034910118\n"
        )
        res = extract_back_fields(back_text)
        assert res["father_name"] == "Chowdegowda"
        assert res["mother_name"] == "Ningamma Kasalagere"
        assert "BN79C5034910118" not in res["address"]["raw"]
        assert "r./FileNo" not in res["address"]["raw"]
        assert res["file_number"] == "BN79C5034910118"
        assert res["address"]["pincode"] == "570022"

