import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Adult Income Prediction")

# User Inputs
age = st.number_input("Age", 18, 100)

workclass = st.selectbox(
    "Workclass",
    encoders['workclass'].classes_
)

education = st.selectbox(
    "Education",
    encoders['education'].classes_
)

marital_status = st.selectbox(
    "Marital Status",
    encoders['marital.status'].classes_
)

occupation = st.selectbox(
    "Occupation",
    encoders['occupation'].classes_
)

sex = st.selectbox(
    "Gender",
    encoders['sex'].classes_
)

capital_gain = st.number_input(
    "Capital Gain",
    0,
    100000
)

capital_loss = st.number_input(
    "Capital Loss",
    0,
    100000
)

hours_per_week = st.number_input(
    "Hours Per Week",
    1,
    100
)

# Encode inputs automatically
input_data = pd.DataFrame({

    'age': [age],

    'workclass': [
        encoders['workclass'].transform([workclass])[0]
    ],

    'education': [
        encoders['education'].transform([education])[0]
    ],

    'marital.status': [
        encoders['marital.status'].transform([marital_status])[0]
    ],

    'occupation': [
        encoders['occupation'].transform([occupation])[0]
    ],

    'sex': [
        encoders['sex'].transform([sex])[0]
    ],

    'capital.gain': [capital_gain],

    'capital.loss': [capital_loss],

    'hours.per.week': [hours_per_week]
})

# Scale
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Income is Greater than 50K")
    else:
        st.error("Income is Less than or Equal to 50K")