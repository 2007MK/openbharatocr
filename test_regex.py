import re
text = """Date of
birth
28 October 1995
National
ity
IND
Status
Visitor - British-
Irish visa scheme
Valid
from
6 August 2026
Valid
until
6 February 2027"""

MONTHS = r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December'
DATE_PAT = (
    r'(?:'
    r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[,.]?\s*'
    r')?'
    r'(?:'
    r'\d{1,2}\s*(?:st|nd|rd|th)?[,\s]+(?:' + MONTHS + r')[a-z]*[,\s]+\d{4}'
    r'|(?:' + MONTHS + r')[a-z]*[,\s]+\d{1,2}[,\s]+\d{4}'
    r'|(?:' + MONTHS + r')[a-z]*[,\s]+\d{1,2}'
    r'|\d{1,2}[\-/](?:' + MONTHS + r')[a-z]*[\-/]\d{2,4}'
    r'|\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}'
    r'|\d{4}[\-/]\d{1,2}[\-/]\d{1,2}'
    r')'
)

SKIP_LABELS = r'(?:book(?:ed|ing)?\s*(?:on|date)?|issued?\s*(?:on|date)?|paid|payment\s*date|printed|generated|date\s*of\s*birth|dob|valid\s*from|valid\s*until|date\s*:|date)'
skip_positions = {
    sm.start(1)
    for sm in re.finditer(r'(?i:' + SKIP_LABELS + r')[\s\S]{0,30}?(' + DATE_PAT + r')', text, re.IGNORECASE)
}

print(skip_positions)
for m in re.finditer(DATE_PAT, text, re.IGNORECASE):
    print("Found:", m.group(0), m.start() in skip_positions)

