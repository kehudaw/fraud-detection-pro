import os, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from .utils import best_threshold_by_cost

DATA_PATH = "data/creditcard.csv"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Missing dataset at data/creditcard.csv")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class","Time"], errors="ignore")
    y = df["Class"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # class imbalance handling
    pos = y_tr.mean()
    scale_pos_weight = (1 - pos) / pos

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist"
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    prob = xgb.predict_proba(X_te)[:,1]
    pr_auc = average_precision_score(y_te, prob)
    thr, savings = best_threshold_by_cost(y_te, prob)

    joblib.dump({"model": xgb, "threshold": thr, "metric": {"pr_auc": pr_auc}, "feature_names": X.columns.tolist()}, "model_xgb.joblib")
    print(f"[XGB] PR-AUC={pr_auc:.4f} | best_threshold={thr:.3f} | savings≈{savings:.1f}")

if __name__ == "__main__":
    main()
