# =========================
# Paths & model loading (FIXED)
# =========================
from pathlib import Path
import joblib
import streamlit as st

# Use the exact filename in your repo. If your file literally has a space, keep it.
MODEL_FILENAME = "artifacts/best model.joblib"      # or "best_model.joblib" if you renamed it

def resolve_model_path() -> Path:
    """
    Resolve the model path robustly for both local runs and Streamlit Cloud.
    If this script is inside the `artifacts/` folder, first try the file next to it.
    Otherwise, try repo-root `artifacts/` and current working directory as fallbacks.
    """
    try:
        script_dir = Path(__file__).parent.resolve()
    except NameError:
        script_dir = Path.cwd().resolve()

    candidates = []

    # 1) If the script itself is inside artifacts/, check sibling file first
    if script_dir.name.lower() == "artifacts":
        candidates.append((script_dir / MODEL_FILENAME).resolve())

    # 2) Repo-root artifacts/ (works even if script moved later)
    repo_root = script_dir if script_dir.name.lower() != "artifacts" else script_dir.parent
    candidates.append((repo_root / "artifacts" / MODEL_FILENAME).resolve())

    # 3) Current working directory fallbacks
    candidates.append((Path.cwd() / MODEL_FILENAME).resolve())
    candidates.append((Path.cwd() / "artifacts" / MODEL_FILENAME).resolve())

    for p in candidates:
        if p.exists():
            return p

    # Return the first candidate for clearer error messaging
    return candidates[0]


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """
    Load and cache the trained pipeline/model from disk.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at: {path}. "
            "If your file is next to this app (inside artifacts/), set MODEL_FILENAME correctly. "
            "Otherwise place the file at 'artifacts/best model.joblib' (or adjust MODEL_FILENAME)."
        )
    return joblib.load(path)


MODEL_PATH = resolve_model_path()

# --- Optional diagnostics to print where the app is looking ---
with st.expander("🔎 Debug: model path resolution", expanded=False):
    st.write("script_dir:", Path(__file__).parent.resolve() if "__file__" in globals() else Path.cwd().resolve())
    st.write("Working directory:", Path.cwd())
    st.write("Resolved MODEL_PATH:", str(MODEL_PATH))

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
