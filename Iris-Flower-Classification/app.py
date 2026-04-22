import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("iris_model.pkl", "rb"))

# Title
st.title("🌸 Iris Flower Classification App")
st.write("Enter flower measurement:")

# Input Fields
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = ["Setosa", "Versicolor", "Verginica"]
    st.success(f"Predicted Species: {species[prediction[0]]}")

