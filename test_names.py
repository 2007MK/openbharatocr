import re
import pprint

raw_text_front = """——— = —"
HIRT WRIST / REPUBLIC OF INDIA
iim cate RNa WORE Pos
IND TIRcTa / INDIAN C6944626
DAS ‘o ‘
cS 9
fe se /¥
RAJU CaN S
- b A
ote © XY °
01/01/1991 ~ AE) Ry
iets se fo
v
TELINAPARA,WEST BENGAL 9 2
. Piace of me S/&
BENGALURU o/s
‘ ; _ g
‘On U D nee @ WR / Dave of tesue a & Be / Date of Expiry &
R 2 03/01/2025 02/01/2035
P< IND DASKKRAJ USK KK KKK KKK KK KKK KKK KKK KEKE KKK KK ¢€
€6944626<91ND9101014M3501025M067214376124<52

CHT LM <a enter 1 type wars Gode woes wane
4, HV aS Nationality 4.1 Passport No.
Pp IND Rca / INDIAN C6944626
. DAS o f
antici —— 4 9
- 2 yawotey J ‘
y = RAJU eel: &,//s
; 7” ~—- . ohh) Chate of Gare he Pye - ara Ors =
- 01/01/1991 " ee a SO/S
=) FT | Place of Garth Ao +
. TELINAPARA,WEST BENGAL 7 /e
, arf ave an Fer / Pace of emu eS &
£
BENGALURU iil 3 is
~ U o% a8 WE BWR / Dave of tswe eet & Byer / Date of Expiry ie
R y D 03/01/2025 02/01/2035
P< INDDAS<<RAI USK KKK KKK KKK KKK KKK KKK KK KEKE KEKE €
' C694 4626<9 1ND9101014M3501025M067214376124<52
"""

def normalize_spaces(text):
    return " ".join(text.split())

lines = [normalize_spaces(line) for line in raw_text_front.splitlines() if line.strip()]
result = {"date_of_birth": "01-01-1991"}
passport_number = "C6944626"

p_idx = -1
d_idx = -1
for i, line in enumerate(lines):
    if passport_number in line:
        p_idx = i
    dob_parts = result["date_of_birth"].split("-")
    if len(dob_parts) == 3:
        dob_slash = f"{dob_parts[0]}/{dob_parts[1]}/{dob_parts[2]}"
        if dob_slash in line or result["date_of_birth"] in line:
            d_idx = i

print("p_idx:", p_idx)
print("d_idx:", d_idx)
print("p_idx line:", lines[p_idx])
print("d_idx line:", lines[d_idx])

name_candidates = []
for line in lines[p_idx + 1:d_idx]:
    words = re.findall(r'\b[A-Z]{2,}\b', line)
    lower_words = re.findall(r'\b[a-z]{3,}\b', line)
    print("examining:", line, words, lower_words)
    if len(words) > 0 and len(lower_words) <= 2:
        ignore_words = {"INDIAN", "REPUBLIC", "INDIA", "CODE", "TYPE", "SURNAME", "NAME", "GIVEN", "PASSPORT", "SEX", "DATE", "BIRTH", "PLACE", "ISSUE", "EXPIRY", "OF", "MAENETS", "MAEN"}
        clean_cands = [w for w in words if w not in ignore_words]
        if clean_cands:
            name_candidates.append(" ".join(clean_cands))

print("name_candidates:", name_candidates)

