import json, os
import joblib, pandas as pd, streamlit as st
import llm, nlp
from common import DEFAULTS, LABEL, OPTIONS, RAW, RANGES, predict_all

st.set_page_config(page_title="Heart Disease AI", page_icon="❤️")

@st.cache_resource
def load():
    if not os.path.exists("models/lr.joblib"):
        import train; train.main()
    return {k: joblib.load(f"models/{k}.joblib") for k in ("lr", "mlp")}, json.load(open("models/metrics.json"))

models, metrics = load()

def default_key():
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY", "")

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("api_key", default_key())
st.session_state.setdefault("nl_text", "55 saal ka aadmi, BP 150, cholesterol 260, max heart rate 120, chest pain during exercise, fasting sugar 130")
st.session_state.setdefault("chat", [])

def fill_form():
    text, found = st.session_state.nl_text, nlp.extract(st.session_state.nl_text)
    if st.session_state.get("use_llm") and st.session_state.api_key:
        found = {**llm.parse_with_llm(st.session_state.api_key, text), **found}  # exact rules win over the LLM
    for k, v in found.items():
        st.session_state[k] = v
    missing = [LABEL[c] for c in RAW if c not in found]
    st.session_state.parse_msg = (f"Filled {len(found)} of {len(RAW)} fields." +
                                  (f" Not found, please check: {', '.join(missing)}." if missing else ""))

def apply_report():
    rep = st.session_state.get("report") or {}
    for k, v in rep.get("fields", {}).items():
        st.session_state[k] = v
    st.session_state.report_fields = list(rep.get("fields", {}))
    st.session_state.report_other = rep.get("other", [])

with st.sidebar:
    st.header("AI settings")
    st.text_input("Anthropic API key", type="password", key="api_key", help="Needed only for the LLM features.")
    lang = st.selectbox("AI reply language", ["English", "Hinglish", "Hindi"])
    st.caption("Educational project, not medical advice.")

st.title("Heart Disease Prediction by Shivani ❤️")
tab_pred, tab_img, tab_nlp, tab_llm, tab_models = st.tabs(
    ["Predict", "Report photo (Vision)", "Describe in words (NLP)", "AI assistant (LLM)", "Models (DL)"])

with tab_pred:
    st.markdown("Enter the details to estimate the likelihood of heart disease.")
    a, b = st.columns(2)
    with a:
        st.slider("Age", *RANGES["Age"], key="Age")
        st.selectbox("Sex", list(OPTIONS["Sex"]), format_func=OPTIONS["Sex"].get, key="Sex")
        st.selectbox("Chest pain type", list(OPTIONS["ChestPainType"]), format_func=OPTIONS["ChestPainType"].get, key="ChestPainType")
        st.slider("Resting blood pressure (mm Hg)", *RANGES["RestingBP"], key="RestingBP")
        st.slider("Cholesterol (mg/dL)", *RANGES["Cholesterol"], key="Cholesterol")
        st.selectbox("Fasting blood sugar > 120 mg/dL", list(OPTIONS["FastingBS"]), format_func=OPTIONS["FastingBS"].get, key="FastingBS")
    with b:
        st.selectbox("Resting ECG", list(OPTIONS["RestingECG"]), format_func=OPTIONS["RestingECG"].get, key="RestingECG")
        st.slider("Maximum heart rate achieved", *RANGES["MaxHR"], key="MaxHR")
        st.selectbox("Exercise-induced angina", list(OPTIONS["ExerciseAngina"]), format_func=OPTIONS["ExerciseAngina"].get, key="ExerciseAngina")
        st.slider("Oldpeak (ST depression)", *RANGES["Oldpeak"], step=0.1, key="Oldpeak")
        st.selectbox("ST slope", list(OPTIONS["ST_Slope"]), format_func=OPTIONS["ST_Slope"].get, key="ST_Slope")
    if st.button("Predict", type="primary", key="predict_btn"):
        row = {c: st.session_state[c] for c in RAW}
        rf = st.session_state.get("report_fields")
        st.session_state.result = {"inputs": row, **predict_all(models, row),
                                   "other": st.session_state.get("report_other", []) if rf is not None else [],
                                   "manual": [LABEL[c] for c in RAW if c not in rf] if rf is not None else []}
        st.session_state.pop("explanation", None)
    r = st.session_state.get("result")
    if r:
        c1, c2, c3 = st.columns(3)
        c1.metric("Logistic Regression", f"{r['lr']:.0%}")
        c2.metric("Neural Network", f"{r['mlp']:.0%}")
        c3.metric("Combined", f"{r['avg']:.0%}")
        st.progress(float(r["avg"]))
        (st.error if r["avg"] >= 0.5 else st.success)(
            "🚨 High risk of heart disease" if r["avg"] >= 0.5 else "✅ Lower risk of heart disease")
        if r["raising"]:
            st.write("**Main factors raising the estimate:** " + ", ".join(r["raising"]))
        if r["lowering"]:
            st.write("**Main factors lowering it:** " + ", ".join(r["lowering"]))
        if abs(r["lr"] - r["mlp"]) > 0.2:
            st.info("The two models disagree noticeably, so treat this estimate with extra caution.")
        st.caption("Not a diagnosis. Please talk to a doctor about your heart health.")

with tab_img:
    st.markdown("Upload a photo or PDF of a medical report (blood test, checkup, ECG report). The AI reads the values, "
                "fills the form, and later explains the result in plain language. A photo of a person cannot show heart risk.")
    st.caption("The file is sent to Anthropic's API to be read. Cover your name and ID number before uploading.")
    up = st.file_uploader("Report photo or PDF", type=["png", "jpg", "jpeg", "webp", "pdf"], key="report_file")
    if not st.session_state.api_key:
        st.info("Add your Anthropic API key in the sidebar to read reports.")
    if up and up.type != "application/pdf":
        st.image(up, width=300)
    too_big = bool(up) and up.size > 5 * 1024 * 1024
    if too_big:
        st.warning("File is larger than 5 MB. Please upload a smaller photo.")
    if st.button("Read report", key="read_btn", disabled=not (up and st.session_state.api_key) or too_big):
        with st.spinner("Reading the report..."):
            st.session_state.report = llm.read_report(st.session_state.api_key, up.getvalue(), up.type)
    rep = st.session_state.get("report")
    if rep and "error" in rep:
        st.error(rep["error"])
    elif rep:
        st.subheader("Values read from the report")
        found = rep["fields"]
        if found:
            st.table(pd.DataFrame({"Value": {LABEL[k]: OPTIONS[k][v] if k in OPTIONS else v for k, v in found.items()}}))
        missing = [LABEL[c] for c in RAW if c not in found]
        if missing:
            st.warning("Not found in the report (enter them yourself in the Predict tab): " + ", ".join(missing))
        if rep["other"]:
            st.markdown("**Other values noticed:**")
            st.table(pd.DataFrame(rep["other"]).rename(columns=str.capitalize))
        st.button("Use these values in the form", on_click=apply_report, key="apply_btn")
        if st.session_state.get("report_fields") is not None:
            st.success("Form filled. Open the Predict tab, fill any missing values, and press Predict.")

with tab_nlp:
    st.markdown("Type the patient details in your own words (English or Hinglish). The app reads the numbers and fills the form.")
    st.text_area("Patient description", key="nl_text", height=120)
    st.checkbox("Also use the LLM for parsing (needs API key)", key="use_llm", disabled=not st.session_state.api_key)
    st.button("Fill the form", on_click=fill_form, key="parse_btn")
    if st.session_state.get("parse_msg"):
        st.success(st.session_state.parse_msg + " Go to the Predict tab to review and predict.")

with tab_llm:
    r = st.session_state.get("result")
    if not st.session_state.api_key:
        st.info("Add your Anthropic API key in the sidebar to use the AI assistant.")
    if r:
        if st.button("Explain my result", key="explain_btn", disabled=not st.session_state.api_key):
            with st.spinner("Thinking..."):
                st.session_state.explanation = llm.explain(st.session_state.api_key, lang, r)
        if st.session_state.get("explanation"):
            st.markdown(st.session_state.explanation)
    else:
        st.caption("Run a prediction first to get a personal explanation.")
    st.divider()
    st.subheader("Ask about heart health")
    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])
    if q := st.chat_input("Ask a question...", disabled=not st.session_state.api_key):
        st.session_state.chat.append({"role": "user", "content": q})
        with st.spinner("Thinking..."):
            reply = llm.chat(st.session_state.api_key, lang, st.session_state.chat[-10:], r)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()

with tab_models:
    st.markdown(f"Both models are trained on the same {metrics['n_train']} patients and tested on {metrics['n_test']} unseen patients.")
    names = {"lr": "Logistic Regression", "mlp": "Neural Network (MLP, 2 hidden layers)", "avg": "Combined (average)"}
    st.dataframe(pd.DataFrame({names[k]: metrics[k] for k in names}).T.rename(columns={
        "accuracy": "Accuracy %", "recall": "Recall %", "f1": "F1 %", "roc_auc": "ROC AUC", "cv_roc_auc": "CV ROC AUC"}))
    st.caption("Recall is the share of sick patients the model catches. Small test sets make these numbers move a lot.")
