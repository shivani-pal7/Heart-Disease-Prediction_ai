"""Rule-based NLP: pull the 11 model inputs out of free text (English + Hinglish). Works offline."""
import re
from common import RANGES

def _first(t, *patterns):
    for p in patterns:
        m = re.search(p, t)
        if m:
            return float(m.group(1))

def extract(text: str) -> dict:
    t, out = text.lower(), {}
    out["Age"] = _first(t, r"(\d{2,3})\s*[- ]?\s*(?:years?|yrs?|yr|saal|sal|वर्ष|साल)(?![a-z])", r"\bage[d]?\D{0,6}(\d{2,3})")
    if re.search(r"\b(female|woman|lady|girl|she|her|wife|mother|aurat|mahila|ladki)\b", t):
        out["Sex"] = "F"
    elif re.search(r"\b(male|man|gentleman|boy|he|his|husband|father|aadmi|mard|purush|ladka)\b", t):
        out["Sex"] = "M"
    out["RestingBP"] = _first(t, r"(?:\bbp\b|blood pressure|b\.p\.?)\D{0,15}?(\d{2,3})")
    out["Cholesterol"] = _first(t, r"(?:cholesterol|cholestrol|chol)\D{0,15}?(\d{2,3})")
    out["MaxHR"] = _first(t, r"(?:max(?:imum)?\.?\s*(?:heart\s*rate|hr|pulse)|heart\s*rate|pulse)\D{0,15}?(\d{2,3})")
    out["Oldpeak"] = _first(t, r"(?:oldpeak|old peak|st depression)[^\d-]{0,10}?(-?\d+(?:\.\d+)?)")
    sugar = _first(t, r"(?:fasting\s*(?:blood\s*)?(?:sugar|glucose)|\bfbs\b|\bsugar\b)\D{0,15}?(\d{2,3})")
    if sugar is not None:
        out["FastingBS"] = int(sugar > 120)
    elif re.search(r"\b(no|non)[- ]diabet", t):
        out["FastingBS"] = 0
    elif "diabet" in t:
        out["FastingBS"] = 1
    pain = r"(?:chest pain|angina|pain|dard|tightness|breathless\w*)"
    effort = r"(?:exercise|exertion|walking|running|climbing|stairs|activity|chalne)"
    if re.search(rf"no (?:chest )?pain (?:during|on|with|while) {effort}|no exercise[- ]induced angina|no angina", t):
        out["ExerciseAngina"] = "N"
    elif re.search(rf"{pain}[^.;\n]{{0,40}}{effort}|{effort}[^.;\n]{{0,40}}{pain}", t):
        out["ExerciseAngina"] = "Y"
    if re.search(r"non[- ]?anginal", t):
        out["ChestPainType"] = "NAP"
    elif "atypical" in t:
        out["ChestPainType"] = "ATA"
    elif re.search(r"typical angina|classic angina", t):
        out["ChestPainType"] = "TA"
    elif re.search(r"no chest pain|without chest pain|no pain|painless|asymptomatic|no symptoms", t):
        out["ChestPainType"] = "ASY"
    elif out.get("ExerciseAngina") == "Y":
        out["ChestPainType"] = "TA"
    if re.search(r"\blvh\b|left ventricular hypertrophy", t):
        out["RestingECG"] = "LVH"
    elif re.search(r"st[- ]?t\b|st abnormal|st wave", t):
        out["RestingECG"] = "ST"
    elif re.search(r"normal (?:ecg|ekg)|(?:ecg|ekg)\s*(?:is\s*)?normal", t):
        out["RestingECG"] = "Normal"
    if re.search(r"down[- ]?slop", t):
        out["ST_Slope"] = "Down"
    elif re.search(r"up[- ]?slop", t):
        out["ST_Slope"] = "Up"
    elif "flat" in t and re.search(r"\bst\b|slope", t):
        out["ST_Slope"] = "Flat"
    clean = {}
    for k, v in out.items():
        if v is None:
            continue
        if k in RANGES and not RANGES[k][0] <= v <= RANGES[k][1]:
            continue
        clean[k] = (v if k == "Oldpeak" else int(v)) if k in RANGES else v
    return clean
