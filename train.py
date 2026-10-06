"""Train Logistic Regression + Neural Network on the same split. Run: python train.py"""
import json, os
import joblib, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from common import RAW, pipelines

def scores(y, p):
    pred = p >= 0.5
    return {"accuracy": round(accuracy_score(y, pred) * 100, 1), "recall": round(recall_score(y, pred) * 100, 1),
            "f1": round(f1_score(y, pred) * 100, 1), "roc_auc": round(roc_auc_score(y, p), 3)}

def main():
    df = pd.read_csv("data/heart.csv")
    X, y = df[RAW], df["HeartDisease"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=42)  # same split as the notebook
    os.makedirs("models", exist_ok=True)
    out, probas = {}, {}
    for name, pipe in pipelines().items():
        cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="roc_auc").mean()
        pipe.fit(Xtr, ytr)
        probas[name] = pipe.predict_proba(Xte)[:, 1]
        out[name] = {**scores(yte, probas[name]), "cv_roc_auc": round(cv, 3)}
        joblib.dump(pipe, f"models/{name}.joblib")
    out["avg"] = scores(yte, (probas["lr"] + probas["mlp"]) / 2)
    out["n_train"], out["n_test"] = len(Xtr), len(Xte)
    json.dump(out, open("models/metrics.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
