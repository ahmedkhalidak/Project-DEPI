import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the scaler and models
scaler = joblib.load('scaler.pkl')
models = {
    "logistic_regression": joblib.load('logistic_regression.pkl'),
    "naive_bayes": joblib.load('naive_bayes.pkl'),
    "svm": joblib.load('svm.pkl'),
    "knn": joblib.load('knn.pkl'),
    "decision_tree": joblib.load('decision_tree.pkl'),
    "random_forest": joblib.load('random_forest.pkl'),
    "xgboost": joblib.load('xgboost.pkl'),
}

# Load the dataset
df = pd.read_csv('cardio_train.csv', delimiter=';')

# Streamlit user interface
st.title("Cardiovascular Health Prediction")
st.write("Enter your details below to predict cardiovascular health:")

# User inputs
age = st.number_input("Age (Years)", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Female", "Male"])
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
ap_hi = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
ap_lo = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=40, max_value=120, value=80)
cholesterol = st.selectbox("Cholesterol Level", options=["Normal", "Above Normal", "Well Above Normal"])
glucose = st.selectbox("Glucose Level", options=["Normal", "Above Normal", "Well Above Normal"])
smoke = st.selectbox("Do you smoke?", options=["Yes", "No"])
alco = st.selectbox("Do you consume alcohol?", options=["Yes", "No"])
active = st.selectbox("Do you exercise?", options=["Yes", "No"])

# Convert inputs
gender = 1 if gender == "Male" else 0
cholesterol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]
glucose = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[glucose]
smoke = 1 if smoke == "Yes" else 0
alco = 1 if alco == "Yes" else 0
active = 1 if active == "Yes" else 0

# Prepare data for prediction, with a dummy value for `id`
input_data = np.array([[0, age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, alco, active]])
input_data_scaled = scaler.transform(input_data)  # تأكد أن input_data يحتوي على 12 ميزة

# Prediction
if st.button("Get Prediction"):
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(input_data_scaled)
        predictions[model_name] = prediction[0]

    st.write("Predictions:")
    for model_name, prediction in predictions.items():
        result = "With Cardiovascular Issues" if prediction == 1 else "Without Cardiovascular Issues"
        st.write(f"{model_name.replace('_', ' ').title()}: {result}")