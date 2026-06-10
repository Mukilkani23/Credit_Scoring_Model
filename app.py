
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 24px 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 24px;
    }
    .header-box h1 { color: #e94560; font-size: 2.2rem; margin: 0 0 6px; }
    .header-box p  { color: #a8a8b3; margin: 0; font-size: 0.95rem; }

    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 8px;
    }

    .result-low {
        background: #f0fff4; border: 2px solid #2ecc71;
        padding: 22px; border-radius: 12px; text-align: center;
    }
    .result-medium {
        background: #fffbf0; border: 2px solid #f39c12;
        padding: 22px; border-radius: 12px; text-align: center;
    }
    .result-high {
        background: #fff5f5; border: 2px solid #e74c3c;
        padding: 22px; border-radius: 12px; text-align: center;
    }
    .result-low h2    { color: #27ae60; margin: 0; }
    .result-medium h2 { color: #e67e22; margin: 0; }
    .result-high h2   { color: #c0392b; margin: 0; }
    .result-sub { font-size: 1rem; color: #555; margin-top: 8px; }

    div[data-testid="stButton"] button {
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Load the saved Pipeline + encoders + feature list.
    @st.cache_resource runs this once and caches — app stays fast.
    """
    base = os.path.dirname(os.path.abspath(__file__))

    pipeline       = joblib.load(os.path.join(base, 'model', 'credit_model_pipeline.pkl'))
    label_encoders = joblib.load(os.path.join(base, 'model', 'label_encoders.pkl'))
    feature_cols   = joblib.load(os.path.join(base, 'model', 'feature_cols.pkl'))

    info = {}
    info_path = os.path.join(base, 'model', 'model_info.txt')
    if os.path.exists(info_path):
        with open(info_path) as f:
            for line in f:
                k, v = line.strip().split('=')
                info[k] = v

    return pipeline, label_encoders, feature_cols, info

try:
    pipeline, label_encoders, feature_cols, model_info = load_artifacts()
    MODEL_NAME = model_info.get('best_model', 'ML Model')
    MODEL_AUC  = model_info.get('roc_auc', 'N/A')
    MODEL_F1   = model_info.get('f1', 'N/A')
    MODEL_ACC  = model_info.get('accuracy', 'N/A')
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    st.info("Run `python train.py` first to generate the model files, then restart the app.")
    st.stop()

# ── Header ────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-box">
    <h1>💳 Credit Scoring Model</h1>
    <p>CodeAlpha ML Internship — Task 1 &nbsp;|&nbsp; Built by Mukilkani R P</p>
</div>
""", unsafe_allow_html=True)

# Model badge row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", MODEL_NAME)
c2.metric("ROC-AUC", MODEL_AUC)
c3.metric("F1-Score", MODEL_F1)
c4.metric("Accuracy", MODEL_ACC)
st.markdown("---")

# ── Feature Glossary (collapsible) ───────────────────────────
with st.expander("📖 Feature Glossary", expanded=False):
    st.markdown("""
| Feature | What it means |
|---------|---------------|
| **Age** | Applicant's age in years |
| **Annual Income** | Yearly income in USD |
| **Home Ownership** | RENT / OWN / MORTGAGE / OTHER |
| **Employment Length** | Years at current employer |
| **Loan Purpose** | Reason for the loan |
| **Loan Grade** | Bank-assigned risk grade. A = safest, G = riskiest |
| **Loan Amount** | Total amount requested |
| **Interest Rate** | Annual % rate offered by the lender |
| **Credit History Length** | Years of credit history on file |
| **Historical Default** | Whether applicant previously defaulted (Y/N) |
""")

# ── Input Form ────────────────────────────────────────────────
st.subheader("📋 Applicant Details")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<p class="section-label">Personal</p>', unsafe_allow_html=True)
    person_age              = st.slider("Age", 18, 75, 30)
    person_income           = st.number_input("Annual Income ($)",
                                              min_value=10_000, max_value=2_000_000,
                                              value=55_000, step=1_000)
    person_home_ownership   = st.selectbox("Home Ownership",
                                           ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    person_emp_length       = st.slider("Employment Length (years)", 0, 40, 5)
    cb_person_default_on_file = st.selectbox("Historical Default on File", ['N', 'Y'])
    cb_person_cred_hist_length = st.slider("Credit History Length (years)", 2, 30, 6)

with col_right:
    st.markdown('<p class="section-label">Loan</p>', unsafe_allow_html=True)
    loan_intent   = st.selectbox("Loan Purpose",
                                 ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL',
                                  'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    loan_grade    = st.selectbox("Loan Grade (A = Best → G = Riskiest)",
                                 ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    loan_amnt     = st.number_input("Loan Amount ($)",
                                    min_value=500, max_value=35_000,
                                    value=10_000, step=500)
    loan_int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 11.0, step=0.5)

    # Live-computed ratio shown as context
    lpi = round(loan_amnt / (person_income + 1), 4)
    dti = round(loan_amnt / (person_income + 1), 4)
    st.metric("Loan-to-Income Ratio (auto)", f"{lpi:.4f}")
    st.caption("Derived automatically from Loan Amount ÷ Annual Income")

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────
predict_btn = st.button("🔍 Assess Credit Risk", use_container_width=True, type="primary")

if predict_btn:

    # ── Compute engineered features (same logic as train.py) ──
    loan_percent_income = loan_amnt / (person_income + 1)
    debt_to_income      = loan_amnt / (person_income + 1)
    is_young            = int(person_age < 30)
    high_interest       = int(loan_int_rate > 15)

    # ── Encode categoricals using saved LabelEncoders ─────────
    try:
        home_enc    = label_encoders['person_home_ownership'].transform([person_home_ownership])[0]
        intent_enc  = label_encoders['loan_intent'].transform([loan_intent])[0]
        grade_enc   = label_encoders['loan_grade'].transform([loan_grade])[0]
        default_enc = label_encoders['cb_person_default_on_file'].transform([cb_person_default_on_file])[0]
    except ValueError as e:
        st.error(f"Encoding error — unexpected value: {e}")
        st.stop()

    # ── Build input DataFrame in exact feature order ───────────
    # Column order MUST match what was used during training.
    # We use feature_cols (saved from train.py) to guarantee this.
    input_df = pd.DataFrame([[
        person_age, person_income, home_enc,
        person_emp_length, intent_enc, grade_enc,
        loan_amnt, loan_int_rate, loan_percent_income,
        default_enc, cb_person_cred_hist_length,
        debt_to_income, is_young, high_interest
    ]], columns=feature_cols)

    # ── Predict using the Pipeline ────────────────────────────
    # The pipeline handles any internal scaling (e.g., for LR).
    # We just pass raw encoded data — pipeline does the rest.
    prediction   = pipeline.predict(input_df)[0]
    prob_default = float(pipeline.predict_proba(input_df)[0][1])
    prob_safe    = 1.0 - prob_default

    # ── Risk Classification ───────────────────────────────────
    if prob_default < 0.30:
        risk_label = "LOW RISK"
        css_class  = "result-low"
        icon       = "✅"
        advice     = "Application looks **favorable**. Low probability of default."
    elif prob_default < 0.60:
        risk_label = "MEDIUM RISK"
        css_class  = "result-medium"
        icon       = "⚠️"
        advice     = "Moderate risk detected. **Additional verification** recommended before approval."
    else:
        risk_label = "HIGH RISK"
        css_class  = "result-high"
        icon       = "🚨"
        advice     = "**High default probability.** Not recommended for approval without strong collateral."

    # ── Result Card ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Risk Assessment Result")

    st.markdown(f"""
    <div class="{css_class}">
        <h2>{icon} {risk_label}</h2>
        <p class="result-sub">Default Probability: <strong>{prob_default * 100:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"\n{advice}")

    # Metric row
    m1, m2, m3 = st.columns(3)
    m1.metric("Default Probability", f"{prob_default * 100:.1f}%",
              delta=f"{(prob_default - 0.22) * 100:+.1f}% vs dataset avg",
              delta_color="inverse")
    m2.metric("Repayment Probability", f"{prob_safe * 100:.1f}%")
    m3.metric("Risk Level", risk_label.split()[0])

    # Probability bar
    st.markdown("**Risk Probability Bar**")
    st.progress(min(prob_default, 1.0))
    st.caption(f"Default: {prob_default*100:.1f}%  |  Safe: {prob_safe*100:.1f}%")

    # ── Key Risk Flags ────────────────────────────────────────
    st.markdown("---")
    st.markdown("**📌 Key Risk Signals Detected**")

    flags = []
    if cb_person_default_on_file == 'Y':
        flags.append("🔴 Prior default on file — strongest negative signal")
    if loan_int_rate > 15:
        flags.append(f"🟠 High interest rate ({loan_int_rate}%) — above 15% threshold")
    if loan_grade in ['E', 'F', 'G']:
        flags.append(f"🟠 Low loan grade ({loan_grade}) — lender already flagged elevated risk")
    if debt_to_income > 0.30:
        flags.append(f"🟠 High debt-to-income ratio ({debt_to_income:.3f})")
    if person_age < 25:
        flags.append(f"🟡 Very young applicant (age {person_age}) — limited credit history likely")
    if loan_grade in ['A', 'B']:
        flags.append(f"🟢 Good loan grade ({loan_grade}) — positive signal")
    if cb_person_cred_hist_length >= 10:
        flags.append(f"🟢 Long credit history ({cb_person_cred_hist_length} yrs) — positive signal")
    if cb_person_default_on_file == 'N' and loan_grade in ['A', 'B', 'C']:
        flags.append("🟢 No historical default + decent grade — strong positive combination")

    if flags:
        for f in flags:
            st.markdown(f"- {f}")
    else:
        st.markdown("- No major risk flags detected")

    # Raw input summary
    with st.expander("🔎 View raw input sent to model", expanded=False):
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.82rem;">
    Built by <b>Mukilkani R P</b> &nbsp;|&nbsp; CodeAlpha ML Internship — Task 1<br>
    Dataset: Credit Risk Dataset (Kaggle) &nbsp;|&nbsp;
    Stack: Python · scikit-learn · XGBoost · Streamlit
</div>
""", unsafe_allow_html=True)
