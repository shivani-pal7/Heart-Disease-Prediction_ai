# Heart Disease AI (ML + DL + NLP + LLM)
- **ML:** Logistic Regression  - **DL:** Neural network (MLP, 2 hidden layers), compared on the same split
- **NLP:** type the patient in words (English/Hinglish); rules extract the 11 inputs
- **LLM:** Claude explains the result, answers questions, and can parse text (needs an API key)
- **Vision:** upload a photo or PDF of a medical report; Claude reads the values, fills the form, and flags other values

## Run
    pip install -r requirements.txt
    python train.py          # trains both models into models/
    streamlit run app.py
    pytest                   # tests

LLM key: paste it in the sidebar, or set `ANTHROPIC_API_KEY`, or use `.streamlit/secrets.toml` (see the example file).
Never commit your real key. Deploy: push to GitHub, then Streamlit Community Cloud, add the key under Secrets.
