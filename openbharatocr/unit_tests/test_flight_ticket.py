import unittest
import openbharatocr

class TestFlightTicketOCR(unittest.TestCase):
    def test_txt_ticket(self):
        res = openbharatocr.flight_ticket('/home/mehul/fx-final/test_ticket.txt')
        self.assertEqual(res.get('pnr'), 'ABC123')
        self.assertIn('JOHN DOE', res.get('passenger_names', []))
        self.assertEqual(res.get('source'), 'Mumbai')
        self.assertEqual(res.get('destination_country'), 'UNITED KINGDOM')
        self.assertEqual(res.get('departure_date'), '12-08-2026')

    def test_1ticket_pdf(self):
        res = openbharatocr.flight_ticket('/home/mehul/fx-final/1ticket.pdf')
        self.assertEqual(res.get('pnr'), '7FZCM7')
        self.assertEqual(res.get('airline_name'), 'LOT Polish Airlines')
        self.assertEqual(res.get('source'), 'Warsaw')
        self.assertEqual(res.get('destination_country'), 'India')
        self.assertEqual(res.get('departure_date'), 'SAT 22 AUGUST 2026')
        self.assertEqual(res.get('return_date'), '')
        self.assertIn('Mr Gangadhar Shankar Umarani', res.get('passenger_names', []))

    def test_2ticket_pdf(self):
        res = openbharatocr.flight_ticket('/home/mehul/fx-final/2ticket.pdf')
        self.assertEqual(res.get('pnr'), '6DICOV')
        self.assertEqual(res.get('airline_name'), 'Air Arabia')
        self.assertEqual(res.get('source'), 'Bengaluru')
        self.assertEqual(res.get('destination_country'), 'Poland')
        self.assertEqual(res.get('departure_date'), '16 Aug 2026')
        self.assertEqual(res.get('return_date'), '')
        self.assertIn('Mr Gangadhar Shankar Umarani', res.get('passenger_names', []))

    def test_deepti_ticket_pdf(self):
        res = openbharatocr.flight_ticket('/home/mehul/openbharatocr/openbharatocr/test_images/deepti_ticket.pdf')
        self.assertEqual(res.get('pnr'), '95W8ZZ')
        self.assertEqual(res.get('airline_name'), 'Etihad Airways')
        self.assertEqual(res.get('source'), 'Bengaluru')
        self.assertEqual(res.get('destination_country'), 'UK')
        self.assertEqual(res.get('departure_date'), '12 Aug 2026')
        self.assertEqual(res.get('return_date'), '31 Aug 2026')
        self.assertIn('Mrs Deepthi Rajesh Kumar Darla', res.get('passenger_names', []))

if __name__ == '__main__':
    unittest.main()
