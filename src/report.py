# src/report.py
import os, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, confusion_matrix

COST_FP = 5.0   # review cost (cheap)
COST_FN = 500.0 # missed-fraud cost (expensive)
DATA_PATH = "data/creditcard.csv"

def eval_at_threshold(y_true, y_prob, thr):
    y_hat = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    cost = fp * COST_FP + fn * COST_FN
    return dict(precision=float(p), recall=float(r), f1=float(f1), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), cost=float(cost))

def main():
    assert os.path.exists(DATA_PATH), "Put creditcard.csv in data/"
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class","Time"], errors="ignore")
    y = df["Class"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # pick best available model
    bundle_path = "model_xgb.joblib" if os.path.exists("model_xgb.joblib") else "model.joblib"
    assert os.path.exists(bundle_path), "Train first: python -m src.train_xgb or python -m src.train_lr"
    bundle = joblib.load(bundle_path)
    model, thr = bundle["model"], float(bundle["threshold"])
    prob = model.predict_proba(X_te)[:,1]

    pr_auc = average_precision_score(y_te, prob)
    stats = eval_at_threshold(y_te, prob, thr)

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.md", "w", encoding="utf-8") as f:
        f.write(f"# Metrics\n\n")
        f.write(f"- **Model**: {'XGBoost' if 'xgb' in bundle_path else 'Logistic Regression'}\n")
        f.write(f"- **PR-AUC**: {pr_auc:.4f}\n")
        f.write(f"- **Operating threshold**: {thr:.3f} (cost_fp={COST_FP}, cost_fn={COST_FN})\n")
        f.write(f"- **Precision/Recall/F1 @ thr**: {stats['precision']:.3f} / {stats['recall']:.3f} / {stats['f1']:.3f}\n")
        f.write(f"- **Confusion matrix @ thr**: TP={stats['tp']} | FP={stats['fp']} | FN={stats['fn']} | TN={stats['tn']}\n")
        f.write(f"- **Estimated cost @ thr**: {stats['cost']:.1f}\n")

    # optional: SHAP beeswarm for XGBoost
    if bundle_path == "model_xgb.joblib":
        try:
            import shap, matplotlib.pyplot as plt
            sample = X_te.sample(min(1000, len(X_te)), random_state=42)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(sample)
            shap.summary_plot(sv, sample, show=False)
            plt.tight_layout()
            plt.savefig("reports/shap_beeswarm.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"[warn] SHAP plot skipped: {e}")

    print("Wrote reports/metrics.md (and shap_beeswarm.png if XGB).")

if __name__ == "__main__":
    main()
