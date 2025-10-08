import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from .utils import best_threshold_by_cost

DATA_PATH = "data/creditcard.csv"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Missing dataset at data/creditcard.csv")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class","Time"], errors="ignore")
    y = df["Class"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=-1))
    ])
    pipe.fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_te)[:,1]
    pr_auc = average_precision_score(y_te, prob)
    thr, savings = best_threshold_by_cost(y_te, prob)

    joblib.dump({"model": pipe, "threshold": thr, "metric": {"pr_auc": pr_auc}}, "model.joblib")
    print(f"[LR] PR-AUC={pr_auc:.4f} | best_threshold={thr:.3f} | savings≈{savings:.1f}")

if __name__ == "__main__":
    main()
