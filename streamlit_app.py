import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="centered",
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .stApp {
        background: #0a0a0f;
        color: #e8e8f0;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00f5a0, #00d9f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .hero-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #555570;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .card {
        background: #12121a;
        border: 1px solid #1e1e2e;
        border-radius: 16px;
        padding: 24px 28px;
        margin: 16px 0;
    }

    .result-legit {
        background: linear-gradient(135deg, #0a1f14, #0d2b1a);
        border: 1.5px solid #00f5a0;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-fraud {
        background: linear-gradient(135deg, #1f0a0a, #2b0d0d);
        border: 1.5px solid #ff4444;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-label {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        margin: 8px 0;
    }

    .prob-bar-wrap {
        background: #1e1e2e;
        border-radius: 999px;
        height: 10px;
        width: 100%;
        margin: 12px 0 4px;
        overflow: hidden;
    }

    .stat-box {
        background: #0d0d15;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }

    .stat-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #00f5a0;
    }

    .stat-label {
        font-size: 0.72rem;
        color: #555570;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }

    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #555570;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #00f5a0, #00d9f5);
        color: #0a0a0f;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 12px;
        padding: 14px 0;
        width: 100%;
        letter-spacing: 0.5px;
        transition: opacity 0.2s;
    }

    div[data-testid="stButton"] > button:hover {
        opacity: 0.85;
        color: #0a0a0f;
    }

    .stNumberInput input, .stSelectbox select {
        background: #0d0d15 !important;
        border: 1px solid #1e1e2e !important;
        color: #e8e8f0 !important;
        font-family: 'Space Mono', monospace !important;
        border-radius: 10px !important;
    }

    hr {
        border-color: #1e1e2e;
        margin: 24px 0;
    }

    .footer {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #333345;
        text-align: center;
        margin-top: 40px;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.getenv("MODEL_PATH", "fraud_detection_model.pkl")
    return joblib.load(path)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Could not load model: {e}")

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ─────────────────────────────────────────
# SAMPLE PRESETS
# ─────────────────────────────────────────
SAMPLE_LEGIT = {
    "Time": 0.0, "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": 2.53634673796914, "V4": 1.37815522427443, "V5": -0.338320769942518,
    "V6": 0.462387777762292, "V7": 0.239598554061257, "V8": 0.0986979012610507,
    "V9": 0.363786969611213, "V10": 0.0907941719789316, "V11": -0.551599533260813,
    "V12": -0.617800855762348, "V13": -0.991389847235408, "V14": -0.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478, "V17": 0.207971241929242,
    "V18": 0.0257905801985591, "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899, "V23": -0.110473910188767,
    "V24": 0.0669280749146731, "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215, "Amount": 149.62
}

SAMPLE_FRAUD = {
    "Time": 406.0, "V1": -2.3122265423263, "V2": 1.95199201064158,
    "V3": -1.60985073229769, "V4": 3.9979055875468, "V5": -0.522187864667764,
    "V6": -1.42654531920595, "V7": -2.53738730624579, "V8": 1.39165724829804,
    "V9": -2.77008927719433, "V10": -2.77227214465915, "V11": 3.20203320709635,
    "V12": -2.89990738849473, "V13": -0.595221881324605, "V14": -4.28925378244217,
    "V15": 0.389724120274487, "V16": -1.14074717980657, "V17": -2.83005567450437,
    "V18": -0.0168224681808257, "V19": 0.416955705037907, "V20": 0.126910559061474,
    "V21": 0.517232370861764, "V22": -0.0350493686052974, "V23": -0.465211076182388,
    "V24": 0.320198198514526, "V25": 0.0445191674731724, "V26": 0.177839798284401,
    "V27": 0.261145002567677, "V28": -0.143275874698919, "Amount": 0.0
}


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown('<div class="hero-title">🛡️ Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-time credit card transaction analysis</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model stats row
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">0.95</div>
        <div class="stat-label">ROC-AUC</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">96%</div>
        <div class="stat-label">Precision</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">284K</div>
        <div class="stat-label">Trained On</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PRESET SELECTOR
# ─────────────────────────────────────────
st.markdown('<div class="section-label">Load a sample transaction</div>', unsafe_allow_html=True)

preset = st.selectbox(
    label="preset",
    options=["— select a preset —", "✅ Sample Legitimate Transaction", "🚨 Sample Fraudulent Transaction"],
    label_visibility="collapsed"
)

if preset == "✅ Sample Legitimate Transaction":
    st.session_state["preset_data"] = SAMPLE_LEGIT
elif preset == "🚨 Sample Fraudulent Transaction":
    st.session_state["preset_data"] = SAMPLE_FRAUD

active = st.session_state.get("preset_data", SAMPLE_LEGIT)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────
st.markdown('<div class="section-label">Transaction details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    time_val = st.number_input("Time (seconds elapsed)", value=float(active["Time"]), format="%.2f")
with col2:
    amount_val = st.number_input("Amount (€)", value=float(active["Amount"]), format="%.2f", min_value=0.0)

st.markdown('<div class="section-label" style="margin-top:16px;">PCA Features (V1 – V28)</div>', unsafe_allow_html=True)
st.caption("These are PCA-transformed features from the original transaction data.")

v_vals = {}
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i - 1) % 4]:
        v_vals[f"V{i}"] = st.number_input(
            f"V{i}",
            value=float(active[f"V{i}"]),
            format="%.4f",
            label_visibility="visible"
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────
if st.button("🔍 Analyze Transaction", use_container_width=True):
    if not model_loaded:
        st.error("Model is not loaded. Cannot make predictions.")
    else:
        input_data = {"Time": time_val, "Amount": amount_val, **v_vals}
        input_array = np.array([[input_data[f] for f in FEATURES]])

        prediction = int(model.predict(input_array)[0])
        probas = model.predict_proba(input_array)[0]
        fraud_prob = float(probas[1])
        legit_prob = float(probas[0])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

        if prediction == 0:
            st.markdown(f"""
            <div class="result-legit">
                <div style="font-size:2.4rem;">✅</div>
                <div class="result-label" style="color:#00f5a0;">Legitimate</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.78rem; color:#555570; margin-top:4px;">
                    Transaction appears safe
                </div>
                <div class="prob-bar-wrap">
                    <div style="background:linear-gradient(90deg,#00f5a0,#00d9f5);
                                height:100%; width:{legit_prob*100:.1f}%; border-radius:999px;"></div>
                </div>
                <div style="font-family:'Space Mono',monospace; font-size:0.8rem; color:#00f5a0;">
                    {legit_prob*100:.1f}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fraud">
                <div style="font-size:2.4rem;">🚨</div>
                <div class="result-label" style="color:#ff4444;">Fraudulent</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.78rem; color:#555570; margin-top:4px;">
                    This transaction is flagged as suspicious
                </div>
                <div class="prob-bar-wrap">
                    <div style="background:linear-gradient(90deg,#ff4444,#ff8800);
                                height:100%; width:{fraud_prob*100:.1f}%; border-radius:999px;"></div>
                </div>
                <div style="font-family:'Space Mono',monospace; font-size:0.8rem; color:#ff4444;">
                    {fraud_prob*100:.1f}% fraud probability
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value" style="color:#00f5a0;">{legit_prob*100:.1f}%</div>
                <div class="stat-label">Legit Probability</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value" style="color:#ff4444;">{fraud_prob*100:.1f}%</div>
                <div class="stat-label">Fraud Probability</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    RANDOM FOREST · 100 ESTIMATORS · TRAINED ON 284,807 TRANSACTIONS<br>
    KAGGLE CREDIT CARD FRAUD DATASET · github.com/bhushanp9/fraud-detection
</div>
""", unsafe_allow_html=True)
