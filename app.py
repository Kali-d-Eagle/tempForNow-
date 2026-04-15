# app.py

import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Predictive Maintenance", layout="centered")

st.title("🏭 Predictive Maintenance System")
st.write("Predict machine failure in advance to reduce downtime.")

st.subheader("🔧 Enter Sensor Values")

volt = st.slider("Voltage", 100, 300, 200)
rotate = st.slider("Rotation", 1000, 2000, 1500)
pressure = st.slider("Pressure", 50, 150, 100)
vibration = st.slider("Vibration", 10, 80, 40)

# simple rolling assumption (for demo)
volt_mean = volt

rotate_mean = rotate
pressure_mean = pressure
vibration_mean = vibration

features = np.array([[volt, rotate, pressure, vibration,
                      volt_mean, rotate_mean, pressure_mean, vibration_mean]])

if st.button("Predict Failure"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Failure! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Machine is Healthy. Probability: {prob:.2f}")