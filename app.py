import streamlit as st
import joblib
import numpy as np

st.title("Fraud Detection App")

st.write("Loading model...")

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

try:
    model = load_model()
    st.success("Model Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

features = []

for i in range(30):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

if st.button("Predict"):
    prediction = model.predict([features])

    if prediction[0] == 1:
        st.error("Fraud Transaction")
    else:
        st.success("Genuine Transaction")
