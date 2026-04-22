import streamlit as st
import pandas as pd 
import numpy as np
import joblib

# Title
st.title("💳 Credit Card Fraud Detection App")

# Load Model
model = joblib.load("fraud_model.pkl")

st.write("Enter transaction details to check if it's Fraud or Genuine")

# Input fields (Simplified)
input_data = []

for i in range(1, 29):
    val = st.number_input(f"V{i}", value = 0.0)
    input_data.append(val)

amount = st.number_input("Amount", value = 0.0)
input_data.append(amount)

# convert to numpy array
input_array = np.array(input_data).reshape(1, -1)

# Predict Button
if st.button("Check Transaction"):
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("🚨 Fradulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")