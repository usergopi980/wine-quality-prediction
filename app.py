import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("wine_model.pkl", "rb"))

st.title("Wine Quality Prediction")

st.write("Enter wine chemical properties")

fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur = st.number_input("Free Sulfur Dioxide")
total_sulfur = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
ph = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

if st.button("Predict"):

    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur,
                            total_sulfur, density, ph, sulphates, alcohol]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Good Quality Wine")
    else:
        st.error("Bad Quality Wine")