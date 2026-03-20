# artifacts/streamlit_app.py

from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# App config (keep at top)
# =========================
st.set_page_config(
    page_title="E‑Commerce Customer Churn Predictor",
    page_icon="🧭",
    layout="centered",
)
st.title("🧭 Customer Churn Predictor")
st.caption("Inputs are aligned to the training schema you provided.")

# =========================
# Paths & model loading
# =========================

# Use the exact filename you committed. If your file is named with a space, keep it.
MODEL_RELATIVE_PATH = Path("artifacts") / "best model.joblib"
# If you renamed to an underscore version, use:
# MODEL_RELATIVE_PATH = Path("artifacts") / "best_model.joblib"


def resolve_model_path() -> Path:
    """
    Resolve the model path robustly so it works locally and on Streamlit Cloud.
    Attempts relative to this script first, then to current working directory.
    """
    try:
        script_dir = Path(__file__).parent.resolve()
    except NameError:
        script_dir = Path.cwd().resolve()

    cand1 = (script_dir / MODEL_RELATIVE_PATH).resolve()
    if cand1.exists():
        return cand1

    cand2 = (Path.cwd() / MODEL_RELATIVE_PATH).resolve()
    if cand2.exists():
        return cand2

    # Return primary candidate for clearer error messages
    return cand1


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """
    Load and cache the trained pipeline/model from disk.
    `st.cache_resource` is ideal for heavy objects like ML pipelines.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at: {path}. "
            "Please ensure your full Pipeline was saved to "
            "'artifacts/best model.joblib' (or update MODEL_RELATIVE_PATH)."
        )
    return joblib.load(path)


MODEL_PATH = resolve_model_path()
try:
    model = load_model(MODEL_PATH)
    st.success(f"✅ Model loaded from: `{MODEL_PATH}`")
except Exception as e:
    st.error("❌ Failed to load model.")
    st.exception(e)
    st.stop()

# =========================
# Columns (schema)
# =========================
INPUT_COLS = [
    "Quantity",
    "Country",
    "Customer_Age",
    "Gender",
    "Marketing_Channel",
    "Category",
    "Subcategory",
    "Discount_Applied",
    "Payment_Method",
    "Promo_Applied",
    "Delivery_Time_Days",
    "Revenue",
    "profit_margin",
    "avg_delivery_days",
]

TARGET_COL = "Churn_Flag"  # not used as input

# =========================
# Utility: scoring that works for many estimators
# =========================
def positive_scores(fitted, X: pd.DataFrame) -> np.ndarray:
    """
    Returns class-1 probabilities for binary classifiers.
    Falls back to decision_function (sigmoid) or raw predictions if needed.
    """
    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(X)[:, 1]
    if hasattr(fitted, "decision_function"):
        z = fitted.decision_function(X)
        # Monotonic mapping; not calibrated but OK as a fallback
        return 1 / (1 + np.exp(-z))
    # Last resort: cast predictions to float
    return fitted.predict(X).astype(float)

# =========================
# CSV upload OR manual entry
# =========================
st.markdown("### Upload a CSV")
st.write(
    "Your CSV should contain **exactly these columns** (order doesn’t matter):  \n"
    f"`{', '.join(INPUT_COLS)}`"
)

uploaded = st.file_uploader("Upload CSV with the required columns", type=["csv"])

if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        st.write("Preview:", df_in.head())

        # Validate columns
        missing = [c for c in INPUT_COLS if c not in df_in.columns]
        extra = [c for c in df_in.columns if c not in INPUT_COLS + [TARGET_COL]]
        if missing:
            st.error(f"Missing required column(s): {missing}")
            st.stop()
        if extra:
            st.info(f"Ignoring extra column(s): {extra}")

        # Keep only inputs in a consistent order (pipeline usually ignores order)
        X = df_in[INPUT_COLS].copy()

        # Predict
        proba = positive_scores(model, X)
        pred = (proba >= 0.5).astype(int)

        out = df_in.copy()  # keep original columns for traceability
        out["y_pred"] = pred
        out["y_proba"] = proba

        st.success("Predictions generated.")
        st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            "Download predictions as CSV",
            data=out.to_csv(index=False),
            file_name="Otim_predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error("Prediction failed. Check column names & types.")
        st.exception(e)

st.markdown("---")
st.markdown("### Or use manual input")

with st.form("manual_form"):
    c1, c2 = st.columns(2)

    # ======= numeric/int inputs =======
    Quantity = c1.number_input("Quantity", min_value=0, value=6, step=1)
    Customer_Age = c2.number_input("Customer_Age", min_value=0.0, value=35.0, step=1.0)
    Discount_Applied = c1.selectbox("Discount_Applied (0/1)", options=[0, 1], index=0)
    Promo_Applied = c2.selectbox("Promo_Applied (0/1)", options=[0, 1], index=0)
    Delivery_Time_Days = c1.number_input("Delivery_Time_Days", min_value=0, value=3, step=1)
    Revenue = c2.number_input("Revenue", min_value=0.0, value=19.90, step=0.10)
    profit_margin = c1.number_input("profit_margin", value=0.30, step=0.01, format="%.3f")
    avg_delivery_days = c2.number_input("avg_delivery_days", min_value=0.0, value=3.0, step=0.1)

    # ======= categoricals =======
    Country = c1.text_input("Country", value="United Kingdom")
    Gender = c2.selectbox("Gender", options=["Female", "Male"], index=0)
    Marketing_Channel = c1.selectbox("Marketing_Channel", options=["Email", "Ads", "Referral", "Organic"], index=0)
    Category = c2.text_input("Category", value="Home Decor")
    Subcategory = c1.text_input("Subcategory", value="Lights")
    Payment_Method = c2.selectbox(
        "Payment_Method", options=["Credit Card", "PayPal", "Cash", "Bank Transfer"], index=0
    )

    submitted = st.form_submit_button("Predict churn")

if submitted:
    row = pd.DataFrame(
        [
            {
                "Quantity": int(Quantity),
                "Country": Country,
                "Customer_Age": float(Customer_Age),
                "Gender": Gender,
                "Marketing_Channel": Marketing_Channel,
                "Category": Category,
                "Subcategory": Subcategory,
                "Discount_Applied": int(Discount_Applied),
                "Payment_Method": Payment_Method,
                "Promo_Applied": int(Promo_Applied),
                "Delivery_Time_Days": int(Delivery_Time_Days),
                "Revenue": float(Revenue),
                "profit_margin": float(profit_margin),
                "avg_delivery_days": float(avg_delivery_days),
            }
        ]
    )[INPUT_COLS]  # ensure column order

    try:
        # Probability & class
        proba = float(positive_scores(model, row)[0])
        if hasattr(model, "predict"):
            pred = int(model.predict(row)[0])
        else:
            pred = int(proba >= 0.5)

        label = "Churn" if pred == 1 else "Retain"
        st.success(f"Prediction: **{label}**  •  Probability: **{proba:.3f}**")
        st.caption(
            "Tip: adjust the threshold (default 0.5) in production to match your retention strategy."
        )
    except Exception as e:
        st.error(
            "Prediction failed. Make sure the model you loaded is the full Pipeline "
            "trained with the same schema."
        )
        st.exception(e)
