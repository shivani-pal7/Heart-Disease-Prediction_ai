"""Shared definitions + model pipelines (the same preprocessing is used for training and prediction)."""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RAW = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG",
       "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
CAT = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]
LABEL = {"Age": "Age", "Sex": "Sex", "ChestPainType": "Chest pain type", "RestingBP": "Resting blood pressure",
         "Cholesterol": "Cholesterol", "FastingBS": "Fasting blood sugar > 120", "RestingECG": "Resting ECG",
         "MaxHR": "Maximum heart rate", "ExerciseAngina": "Exercise-induced angina",
         "Oldpeak": "Oldpeak (ST depression)", "ST_Slope": "ST slope"}
OPTIONS = {"Sex": {"M": "Male", "F": "Female"},
           "ChestPainType": {"ATA": "Atypical angina (ATA)", "NAP": "Non-anginal pain (NAP)",
                             "ASY": "No symptoms (ASY)", "TA": "Typical angina (TA)"},
           "FastingBS": {1: "Yes", 0: "No"},
           "RestingECG": {"Normal": "Normal", "ST": "ST-T abnormality", "LVH": "Left ventricular hypertrophy"},
           "ExerciseAngina": {"Y": "Yes", "N": "No"},
           "ST_Slope": {"Up": "Up", "Flat": "Flat", "Down": "Down"}}
RANGES = {"Age": (18, 100), "RestingBP": (80, 220), "Cholesterol": (100, 600), "MaxHR": (60, 220), "Oldpeak": (0.0, 6.0)}
DEFAULTS = {"Age": 50, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 120, "Cholesterol": 200, "FastingBS": 0,
            "RestingECG": "Normal", "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 1.0, "ST_Slope": "Up"}

def zeros_to_nan(X):
    return np.where(np.asarray(X, dtype=float) == 0, np.nan, X)  # 0 BP / 0 cholesterol = missing in this dataset

def _pre():
    fix = Pipeline([("zero", FunctionTransformer(zeros_to_nan, feature_names_out="one-to-one")),
                    ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    return ColumnTransformer([("fix", fix, ["RestingBP", "Cholesterol"]),
                              ("num", StandardScaler(), ["Age", "MaxHR", "Oldpeak"]),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

def pipelines():
    return {"lr": Pipeline([("pre", _pre()), ("clf", LogisticRegression(max_iter=1000))]),
            "mlp": Pipeline([("pre", _pre()), ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, validation_fraction=0.15,
                max_iter=1000, random_state=42))])}

def factors(pipe, df, top=3):
    """Main drivers from the logistic regression: coefficient x input value, grouped per original feature."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    x = pre.transform(df)
    x = x.toarray()[0] if hasattr(x, "toarray") else x[0]
    contrib = {}
    for name, v, w in zip(pre.get_feature_names_out(), x, clf.coef_[0]):
        col = name.split("__", 1)[1]
        key = next(c for c in RAW if col == c or col.startswith(c + "_"))
        contrib[key] = contrib.get(key, 0) + v * w
    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    return ([LABEL[k] for k, v in ranked if v > 0][:top], [LABEL[k] for k, v in ranked[::-1] if v < 0][:top])

def predict_all(models, row: dict) -> dict:
    df = pd.DataFrame([row])[RAW]
    lr, mlp = (float(models[k].predict_proba(df)[0, 1]) for k in ("lr", "mlp"))
    up, down = factors(models["lr"], df)
    return {"lr": lr, "mlp": mlp, "avg": (lr + mlp) / 2, "raising": up, "lowering": down}
