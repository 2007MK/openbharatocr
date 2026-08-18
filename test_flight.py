import openbharatocr
from openbharatocr.ocr.flight_ticket import extract_details_from_text
text = """
Booking Reference PNR: AB12C3
Passenger Name: Mr. Rahul Sharma
Ms. Aditi Singh
Flight Details: 6E 123
Departure: 12-Oct-2023
Return: 15-Oct-2023
Duration: 2h 30m
"""
print(extract_details_from_text(text))
