# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "fraud_detection_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except:
    model = None

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details below to check whether the transaction is Fraudulent or Genuine.")

# ---------------------------
# Feature Names
# ---------------------------
features = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
    'V28', 'Amount'
]

# ---------------------------
# User Input
# ---------------------------
input_data = []

col1, col2, col3 = st.columns(3)

for i, feature in enumerate(features):

    with [col1, col2, col3][i % 3]:
        value = st.number_input(
            f"{feature}",
            value=0.0,
            step=0.1
        )
        input_data.append(value)

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔍 Predict Transaction"):

    if model is None:
        st.error("Model file not found. Please place fraud_detection_model.pkl in the same folder.")
    else:

        input_array = np.array(input_data).reshape(1, -1)

        try:
            prediction = model.predict(input_array)

            if prediction[0] == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Genuine Transaction")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📌 Instructions")
st.sidebar.write("""
1. Enter all transaction feature values.
2. Click on Predict Transaction.
3. The app will classify the transaction as:
   - Genuine
   - Fraudulent
""")

st.sidebar.info("Built using Streamlit + Machine Learning")
