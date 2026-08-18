from openbharatocr.ocr.flight_ticket import ticket
import json

print("1TICKET")
print(json.dumps(ticket("/home/mehul/fx-final/1ticket.pdf"), indent=2))
print("2TICKET")
print(json.dumps(ticket("/home/mehul/fx-final/2ticket.pdf"), indent=2))
