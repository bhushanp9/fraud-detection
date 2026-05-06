# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    model = joblib.load("fraud_detection_model.pkl")
    return model

model = load_model()

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stButton>button {
    width: 100%;
    background-color: #00C853;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
}

.stButton>button:hover {
    background-color: #00E676;
    color: black;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================
st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Detect whether a transaction is Fraudulent or Genuine using Machine Learning")

st.divider()

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("📌 About")
st.sidebar.info(
    """
    This application predicts whether a credit card transaction is:

    ✅ Genuine Transaction  
    🚨 Fraudulent Transaction

    Built using:
    - Streamlit
    - Scikit-Learn
    - Machine Learning
    """
)

st.sidebar.title("⚙ Instructions")
st.sidebar.write("""
1. Enter all feature values.
2. Click on Predict.
3. View prediction result.
""")

# =========================================
# FEATURE INPUTS
# =========================================

features = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
    'V28', 'Amount'
]

st.subheader("📝 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

input_data = []

for i, feature in enumerate(features):

    with [col1, col2, col3][i % 3]:
        value = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )
        input_data.append(value)

st.divider()

# =========================================
# PREDICT BUTTON
# =========================================
if st.button("🔍 Predict Transaction"):

    input_array = np.array(input_data).reshape(1, -1)

    try:
        prediction = model.predict(input_array)

        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:

            st.markdown("""
            <div class='result-box' style='background-color:#ffebee; color:#d32f2f;'>
                🚨 Fraudulent Transaction Detected
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='result-box' style='background-color:#e8f5e9; color:#2e7d32;'>
                ✅ Genuine Transaction
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# =========================================
# FOOTER
# =========================================
st.divider()

st.markdown(
    """
    <center>
        <h4>💡 Machine Learning Fraud Detection App</h4>
        <p>Built with Streamlit</p>
    </center>
    """,
    unsafe_allow_html=True
)
