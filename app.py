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
    border: none;
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
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================
st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
### Machine Learning Based Fraud Detection App

This application predicts whether a transaction is:

- ✅ Genuine Transaction
- 🚨 Fraudulent Transaction
""")

st.divider()

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("📌 About Project")

st.sidebar.info("""
Fraud Detection System using Machine Learning.

Technologies Used:
- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
""")

st.sidebar.title("⚙ Instructions")

st.sidebar.write("""
### Manual Prediction
1. Enter transaction values
2. Click Predict Transaction

### CSV Prediction
1. Upload CSV file
2. Click Predict Uploaded Data
3. Download prediction results
""")

# =========================================
# FEATURES
# =========================================
features = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
    'V28', 'Amount'
]

# =========================================
# MANUAL INPUT SECTION
# =========================================
st.subheader("📝 Manual Transaction Prediction")

col1, col2, col3 = st.columns(3)

input_data = []

for i, feature in enumerate(features):

    with [col1, col2, col3][i % 3]:

        value = st.number_input(
            label=feature,
            value=0.0,
            format="%.6f"
        )

        input_data.append(value)

# =========================================
# MANUAL PREDICTION
# =========================================
if st.button("🔍 Predict Transaction"):

    # Validation
    if all(value == 0.0 for value in input_data):

        st.warning("⚠ Please enter transaction values before prediction.")

    else:

        try:

            # Convert to numpy array
            input_array = np.array(input_data).reshape(1, -1)

            # Predict
            prediction = model.predict(input_array)

            st.subheader("📊 Prediction Result")

            # Fraud
            if prediction[0] == 1:

                st.markdown("""
                <div class='result-box'
                style='background-color:#ffebee; color:#d32f2f;'>

                🚨 Fraudulent Transaction Detected

                </div>
                """, unsafe_allow_html=True)

            # Genuine
            else:

                st.markdown("""
                <div class='result-box'
                style='background-color:#e8f5e9; color:#2e7d32;'>

                ✅ Genuine Transaction

                </div>
                """, unsafe_allow_html=True)

        except Exception as e:

            st.error(f"Prediction Error: {e}")

# =========================================
# CSV FILE PREDICTION
# =========================================
st.divider()

st.subheader("📂 Bulk Prediction Using CSV File")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # Read CSV
        df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Dataset")
        st.dataframe(df.head())

        st.write(f"Dataset Shape: {df.shape}")

        # Predict Button
        if st.button("🚀 Predict Uploaded Data"):

            predictions = model.predict(df)

            df["Prediction"] = predictions

            df["Prediction"] = df["Prediction"].map({
                0: "Genuine",
                1: "Fraud"
            })

            st.success("✅ Prediction Completed Successfully")

            # Metrics
            fraud_count = (df["Prediction"] == "Fraud").sum()
            genuine_count = (df["Prediction"] == "Genuine").sum()

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric("🚨 Fraud Transactions", fraud_count)

            with metric2:
                st.metric("✅ Genuine Transactions", genuine_count)

            # Show Results
            st.write("### Prediction Results")
            st.dataframe(df)

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="fraud_prediction_results.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(f"Error Processing File: {e}")

# =========================================
# FOOTER
# =========================================
st.divider()

st.markdown("""
<center>

<h4>💡 Fraud Detection Using Machine Learning</h4>

<p>Built with Streamlit</p>

</center>
""", unsafe_allow_html=True)
