import streamlit as st
import pandas as pd
import joblib

# ---------- Load model ----------
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---------- Page config ----------
st.set_page_config(
    page_title="Banking Churn Predictor",
    page_icon="🏦",
    layout="wide",
)

# ---------- Custom CSS: black & deep gold ----------
st.markdown("""
<style>
    /* Page background */
    .stApp { background: #0a0a0a; color: #f5f5f5; }
    header[data-testid="stHeader"] { background: #0a0a0a; }
    h1, h2, h3, p, label, span, div { color: #f5f5f5; }

    /* Title — brighter gold with a soft glow */
    .big-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #d4af37, #f5d76e, #fff2b3, #f5d76e, #d4af37);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.02em;
        margin-bottom: 0;
        filter: drop-shadow(0 0 14px rgba(245, 215, 110, 0.45));
    }
    .subtitle {
        color: #a3a3a3 !important;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    /* Section labels */
    .section-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #d4af37 !important;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 0.35rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }

    /* ---------- Bordered containers (Streamlit 1.64 selector) ---------- */
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"]) {
        background: #121212;
        border: 1px solid #4a3a12 !important;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 6px 24px rgba(212, 175, 55, 0.06);
    }

    /* ---------- LEFT column container: strong golden glow ---------- */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1)
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"]) {
        border: 1px solid #f5d76e !important;
        box-shadow:
            0 0 0 1px rgba(245, 215, 110, 0.55),
            0 0 18px rgba(245, 215, 110, 0.45),
            0 0 40px rgba(212, 175, 55, 0.55),
            0 0 80px rgba(212, 175, 55, 0.30),
            inset 0 0 24px rgba(245, 215, 110, 0.08),
            0 8px 30px rgba(0, 0, 0, 0.85) !important;
    }

    /* ---------- RIGHT column container: main visual focus ---------- */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2)
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"]) {
        background:
            radial-gradient(circle at 50% 0%, rgba(212, 175, 55, 0.15), transparent 70%),
            linear-gradient(180deg, #2a1f08 0%, #1a1406 100%) !important;
        border: 2px solid #f5d76e !important;
        box-shadow:
            0 0 0 1px rgba(245, 215, 110, 0.5),
            0 0 24px rgba(245, 215, 110, 0.45),
            0 0 60px rgba(212, 175, 55, 0.40),
            0 18px 50px rgba(0, 0, 0, 0.95) !important;
    }
    /* All text inside the right panel in cream */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2)
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"]) * {
        color: #f5f0dc;
    }

    /* Right panel section label — brighter gold + stronger rule */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2)
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"])
        .section-label {
        color: #f5d76e !important;
        border-bottom: 1px solid rgba(245, 215, 110, 0.35);
        font-size: 0.9rem;
        letter-spacing: 0.18em;
    }

    /* ---------- DEEP GOLD SLIDER ---------- */
    .stSlider [data-baseweb="slider"] [role="progressbar"],
    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #a67c00 !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #a67c00 !important;
        border-color: #a67c00 !important;
        box-shadow: 0 0 0 4px rgba(166, 124, 0, 0.25) !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] > div {
        background-color: #a67c00 !important;
        color: #0a0a0a !important;
        font-weight: 700;
    }
    .stSlider [data-testid="stTickBar"] { color: #a3a3a3 !important; }

    /* ---------- Selectbox / number input ---------- */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        background-color: #171717 !important;
        color: #f5f5f5 !important;
        border: 1px solid #2a2a2a !important;
    }

    /* ---------- Deep gold primary button ---------- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #a67c00, #d4af37);
        color: #0a0a0a;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        letter-spacing: 0.03em;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #d4af37, #a67c00);
        color: #0a0a0a;
    }

    /* ---------- Progress bar (gold) ---------- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #a67c00, #f5d76e) !important;
    }
    .stProgress > div > div > div {
        background-color: rgba(245, 215, 110, 0.15) !important;
    }

    /* ---------- Result card (on top of the gold panel) ---------- */
    .result-card {
        padding: 1.8rem 1.2rem;
        border-radius: 16px;
        text-align: center;
        background: rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(245, 215, 110, 0.35);
        box-shadow: inset 0 0 30px rgba(245, 215, 110, 0.08);
    }
    .result-high  { border: 1px solid #f87171; box-shadow: 0 0 30px rgba(248,113,113,0.35), inset 0 0 30px rgba(248,113,113,0.08); }
    .result-mid   { border: 1px solid #facc15; box-shadow: 0 0 30px rgba(250,204,21,0.35), inset 0 0 30px rgba(250,204,21,0.08); }
    .result-low   { border: 1px solid #4ade80; box-shadow: 0 0 30px rgba(74,222,128,0.35), inset 0 0 30px rgba(74,222,128,0.08); }

    .result-prob {
        font-size: 4rem;
        font-weight: 900;
        margin: 0.3rem 0;
        color: #f5d76e;
        letter-spacing: -0.02em;
        text-shadow: 0 0 24px rgba(245, 215, 110, 0.45);
    }
    .result-high .result-prob { color: #f87171; text-shadow: 0 0 24px rgba(248,113,113,0.5); }
    .result-mid  .result-prob { color: #facc15; text-shadow: 0 0 24px rgba(250,204,21,0.5); }
    .result-low  .result-prob { color: #4ade80; text-shadow: 0 0 24px rgba(74,222,128,0.5); }

    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #c9b88a !important;
        margin-bottom: 0.5rem;
    }
    .verdict {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f5f0dc !important;
        margin-top: 0.6rem;
    }

    /* ---------- Alerts / expander ---------- */
    .stAlert {
        background-color: #171717 !important;
        border: 1px solid #2a2a2a !important;
    }
    .stAlert * { color: #e5e5e5 !important; }

    details {
        background-color: #171717 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
    }
    details summary { color: #d4af37 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="big-title">🏦 Banking Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate a customer\'s likelihood to leave the bank.</div>', unsafe_allow_html=True)

# ---------- Two columns ----------
left, right = st.columns([2, 1], gap="large")

# ===== LEFT: inputs =====
with left:
    with st.container(border=True):
        st.markdown('<div class="section-label">👤 Personal</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 92, 40)
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c2:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            credit_score = st.slider("Credit Score", 300, 850, 650)

        st.markdown('<div class="section-label">💰 Financial</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            balance = st.number_input("Balance", 0.0, 300000.0, 50000.0, step=1000.0)
        with c4:
            salary = st.number_input("Estimated Salary", 0.0, 300000.0, 100000.0, step=1000.0)

        st.markdown('<div class="section-label">🏛️ Account</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            tenure = st.slider("Tenure (years)", 0, 10, 5)
            num_products = st.slider("Products", 1, 4, 2)
        with c6:
            has_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
            is_active = st.selectbox("Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No")

        st.write("")
        predict_clicked = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

# ===== RIGHT: prediction =====
with right:
    with st.container(border=True):
        st.markdown('<div class="section-label">📊 Prediction</div>', unsafe_allow_html=True)

        if predict_clicked:
            row = {
                "CreditScore": credit_score,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_card,
                "IsActiveMember": is_active,
                "EstimatedSalary": salary,
                "Geography_Germany": 1 if geography == "Germany" else 0,
                "Geography_Spain": 1 if geography == "Spain" else 0,
                "Gender_Male": 1 if gender == "Male" else 0,
            }
            X_input = pd.DataFrame([row])[model_columns]
            proba = model.predict_proba(X_input)[0, 1]
            pct = proba * 100

            if proba >= 0.5:
                css_class, verdict = "result-high", "⚠️ High risk of churn"
            elif proba >= 0.3:
                css_class, verdict = "result-mid", "⚡ Moderate risk of churn"
            else:
                css_class, verdict = "result-low", "✅ Low risk of churn"

            st.markdown(f"""
            <div class="result-card {css_class}">
                <div class="result-label">Churn probability</div>
                <div class="result-prob">{pct:.1f}%</div>
                <div class="verdict">{verdict}</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.progress(min(int(pct), 100))

            with st.expander("See input details"):
                st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)
        else:
            st.info("Set the profile on the left, then click **Predict Churn**.")