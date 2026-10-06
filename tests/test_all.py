import os
import pytest
from streamlit.testing.v1 import AppTest
import llm, nlp

def test_nlp_english_and_hinglish():
    o = nlp.extract("55 year old male, BP 150/90, cholesterol 260, max heart rate 120, chest pain during exercise, "
                    "fasting sugar 130, normal ECG, flat ST slope, oldpeak 2.5")
    assert o == {"Age": 55, "Sex": "M", "RestingBP": 150, "Cholesterol": 260, "MaxHR": 120, "Oldpeak": 2.5,
                 "FastingBS": 1, "ExerciseAngina": "Y", "ChestPainType": "TA", "RestingECG": "Normal", "ST_Slope": "Flat"}
    h = nlp.extract("42 saal ki aurat, no chest pain, BP 118")
    assert h["Sex"] == "F" and h["Age"] == 42 and h["ChestPainType"] == "ASY" and h["RestingBP"] == 118

def test_nlp_rejects_out_of_range():
    assert "Age" not in nlp.extract("age 250") and "RestingBP" not in nlp.extract("bp 20")

def test_llm_parse_validates(monkeypatch):
    monkeypatch.setattr(llm, "_complete", lambda *a, **k: '```json\n{"Age": 61, "Sex": "F", "ChestPainType": "XXX", "MaxHR": 999}\n```')
    assert llm.parse_with_llm("k", "text") == {"Age": 61, "Sex": "F"}
    monkeypatch.setattr(llm, "_complete", lambda *a, **k: "sorry, no json")
    assert llm.parse_with_llm("k", "text") == {}

def run():
    return AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=90).run()

def test_app_runs_and_categoricals_matter():
    assert not run().exception
    probs = {}
    for key, val in [("base", None), ("female", ("Sex", "F")), ("asy", ("ChestPainType", "ASY")), ("angina", ("ExerciseAngina", "Y"))]:
        at = run()
        if val:
            at.session_state[val[0]] = val[1]
        at.button(key="predict_btn").click().run()
        assert not at.exception
        probs[key] = at.session_state["result"]["avg"]
    assert len({round(p, 4) for p in probs.values()}) == 4, probs

def test_nlp_tab_fills_form():
    at = run()
    at.text_area(key="nl_text").set_value("60 year old male BP 140 cholesterol 230 no chest pain").run()
    at.button(key="parse_btn").click().run()
    assert not at.exception and at.session_state["Age"] == 60 and at.session_state["ChestPainType"] == "ASY"

def test_explain_flow(monkeypatch):
    monkeypatch.setattr(llm, "_complete", lambda *a, **k: "Plain explanation")
    at = run(); at.session_state["api_key"] = "sk-test"
    at.button(key="predict_btn").click().run()
    at.button(key="explain_btn").click().run()
    assert not at.exception and at.session_state["explanation"] == "Plain explanation"

def test_read_report_parses_and_validates(monkeypatch):
    payload = ('{"fields": {"Age": 58, "Cholesterol": 310, "Sex": "X"}, "other": [{"name":"HDL","value":"32","unit":"mg/dL","status":"low"},'
               '{"name":"Odd","value":"1","unit":"","status":"weird"}, "junk"]}')
    monkeypatch.setattr(llm, "_complete", lambda *a, **k: payload)
    r = llm.read_report("k", b"x", "image/png")
    assert r["fields"] == {"Age": 58, "Cholesterol": 310}
    assert [o["status"] for o in r["other"]] == ["low", "unknown"]

def test_read_report_errors(monkeypatch):
    monkeypatch.setattr(llm, "_complete", lambda *a, **k: "⚠️ The API key was rejected.")
    assert "error" in llm.read_report("k", b"x", "image/png")
    monkeypatch.setattr(llm, "_complete", lambda *a, **k: "no json here")
    assert "error" in llm.read_report("k", b"x", "application/pdf")

def test_report_flow_reaches_explanation(monkeypatch):
    seen = []
    def fake(api_key, system, messages, max_tokens=600):
        seen.append(messages[-1]["content"]); return "ok"
    monkeypatch.setattr(llm, "_complete", fake)
    at = run(); at.session_state["api_key"] = "sk-test"
    at.session_state["report"] = {"fields": {"Age": 58, "Cholesterol": 310},
                                  "other": [{"name": "HDL", "value": "32", "unit": "mg/dL", "status": "low"}]}
    at.run()
    at.button(key="apply_btn").click().run()
    assert not at.exception and at.session_state["Age"] == 58
    at.button(key="predict_btn").click().run()
    at.button(key="explain_btn").click().run()
    assert "HDL 32 mg/dL (low)" in seen[-1] and "Resting blood pressure" in seen[-1]
