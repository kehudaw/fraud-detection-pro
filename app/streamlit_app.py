# app/streamlit_app.py  (replace the file with this)
import os, joblib, pandas as pd, numpy as np, streamlit as st

st.set_page_config(page_title="Fraud Detection (Pro)", page_icon="🛡️", layout="wide")
st.title("🛡️ Fraud Detection By Matthew Jake Inson (Pro)")
st.caption("Use real rows from the dataset to fill V1–V28 + Amount. If XGBoost is loaded, SHAP explains the decision.")

MODEL_XGB = "model_xgb.joblib"
MODEL_LR = "model.joblib"
DATA_PATH = "data/creditcard.csv"

# Load model (prefer XGB)
if os.path.exists(MODEL_XGB):
    bundle = joblib.load(MODEL_XGB); model_type = "xgb"
elif os.path.exists(MODEL_LR):
    bundle = joblib.load(MODEL_LR); model_type = "lr"
else:
    st.error("No model found. Run `python -m src.train_xgb` or `python -m src.train_lr` first.")
    st.stop()

model = bundle["model"]
thr = float(bundle["threshold"])
pr_auc = bundle.get("metric", {}).get("pr_auc")
feature_names = bundle.get("feature_names", [f"V{i}" for i in range(1,29)] + ["Amount"])

col1, col2 = st.columns(2)
with col1: st.metric("Optimal threshold", f"{thr:.3f}")
with col2:
    if pr_auc is not None: st.metric("Test PR-AUC", f"{pr_auc:.4f}")

# ---- Data helpers
@st.cache_data
def load_data(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None

df = load_data(DATA_PATH)

st.sidebar.header("Input mode")
if df is not None:
    mode = st.sidebar.radio(
        "Choose how to fill inputs:",
        ["Manual entry", "Random LEGIT sample", "Random FRAUD sample", "Pick row by index"],
        index=1
    )
else:
    mode = "Manual entry"
    st.sidebar.info("Dataset not found at data/creditcard.csv — using manual entry only.")

true_label = None
inputs = {}

# Prefill from dataset if requested
if df is not None and mode != "Manual entry":
    if mode == "Random LEGIT sample":
        row = df[df["Class"] == 0].sample(1, random_state=None).iloc[0]
    elif mode == "Random FRAUD sample":
        row = df[df["Class"] == 1].sample(1, random_state=None).iloc[0]
    else:  # Pick row by index
        max_idx = len(df) - 1
        idx = st.sidebar.number_input("Row index (0..{}):".format(max_idx), min_value=0, max_value=max_idx, value=0, step=1)
        row = df.iloc[int(idx)]
    true_label = int(row["Class"]) if "Class" in row else None
    for c in feature_names:
        inputs[c] = float(row[c]) if c in row.index else 0.0

# Manual entry (or allow edits to prefilled values)
st.subheader("Enter / review features")
with st.form("inference"):
    cols_per_row = 4
    keys = feature_names
    for i in range(0, len(keys), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, c in enumerate(keys[i:i+cols_per_row]):
            default_val = inputs.get(c, 0.0)
            with row_cols[j]:
                inputs[c] = st.number_input(c, value=float(default_val), step=0.1, format="%.4f")
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([inputs])[feature_names]
    prob = float(model.predict_proba(X)[:, 1][0])
    decision = "Fraud" if prob >= thr else "Legit"

    left, right = st.columns([2,1])
    with left:
        st.success(f"Fraud probability: **{prob:.3f}**  →  **{decision}**")
    with right:
        if true_label is not None:
            st.caption(f"True label of sampled row: **{true_label}** (1=Fraud, 0=Legit)")

    # Explain (XGB only)
    if model_type == "xgb":
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            shap_vals = sv[0] if isinstance(sv, list) else sv
            contrib = pd.DataFrame({
                "feature": feature_names,
                "shap_value": shap_vals[0],
                "abs_shap": np.abs(shap_vals[0]),
                "value": [inputs[c] for c in feature_names]
            }).sort_values("abs_shap", ascending=False).head(12)
            st.subheader("Top contributing features (SHAP)")
            st.dataframe(contrib[["feature","value","shap_value"]].reset_index(drop=True))
            st.caption("Positive SHAP pushes toward 'Fraud'; negative toward 'Legit'.")
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
    else:
        st.info("Load the XGBoost model (`python -m src.train_xgb`) to see SHAP explanations.")
