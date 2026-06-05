import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Adult Income Prediction")

age = st.number_input("Age", 18, 100)

workclass = st.selectbox("Workclass", [
    "Private",
    "State Government",
    "Local Government",
    "Federal Government",
    "Self Employed",
    "Without Pay"
])

education = st.selectbox("Education", [
    "Secondary Education",
    "Higher Secondary Education",
    "Bachelors",
    "Masters",
    "Doctorate",
    "Other"
])

marital_status = st.selectbox("Marital Status", [
    "Married",
    "Single",
    "Divorced",
    "Separated",
    "Widowed"
])

occupation = st.selectbox("Occupation", [
    "Sales",
    "Professor",
    "Armed Forces",
    "Farmer",
    "Craft Repair",
    "Handlers-Cleaners",
    "Engineer",
    "Other Service"
])

relationship = st.selectbox("Relationship", [
    "Husband",
    "Not-in-family",
    "Wife",
    "Unmarried"
])

sex = st.selectbox(
    "Gender",
    encoders['sex'].classes_
)

capital_gain = st.number_input("Capital Gain", 0, 100000)

capital_loss = st.number_input("Capital Loss", 0, 100000)

hours_per_week = st.number_input("Hours Per Week", 1, 100)

workclass_dict = {
    "Private": "Private",
    "State Government": "State-gov",
    "Local Government": "Local-gov",
    "Federal Government": "Federal-gov",
    "Self Employed": "Self-emp-not-inc",
    "Without Pay": "Without-pay"
}

education_dict = {
    "Secondary Education": "10th",
    "Higher Secondary Education": "HS-grad",
    "Bachelors": "Bachelors",
    "Masters": "Masters",
    "Doctorate": "Doctorate",
    "Other": "Some-college"
}

marital_dict = {
    "Married": "Married-civ-spouse",
    "Single": "Never-married",
    "Divorced": "Divorced",
    "Separated": "Separated",
    "Widowed": "Widowed"
}

occupation_dict = {
    "Sales": "Sales",
    "Professor": "Prof-specialty",
    "Armed Forces": "Armed-Forces",
    "Farmer": "Farming-fishing",
    "Craft Repair": "Craft-repair",
    "Handlers-Cleaners": "Handlers-cleaners",
    "Engineer": "Tech-support",
    "Other Service": "Other-service"
}

relationship_dict = {
    "Husband": "Husband",
    "Not-in-family": "Not-in-family",
    "Wife": "Wife",
    "Unmarried": "Unmarried"
}

workclass = workclass_dict[workclass]
education = education_dict[education]
marital_status = marital_dict[marital_status]
occupation = occupation_dict[occupation]
relationship = relationship_dict[relationship]

input_data = pd.DataFrame({

    "age": [age],

    "workclass": [
        encoders['workclass'].transform([workclass])[0]
    ],

    "education": [
        encoders['education'].transform([education])[0]
    ],

    "marital.status": [
        encoders['marital.status'].transform([marital_status])[0]
    ],

    "occupation": [
        encoders['occupation'].transform([occupation])[0]
    ],

    "relationship": [
        encoders['relationship'].transform([relationship])[0]
    ],

    "sex": [
        encoders['sex'].transform([sex])[0]
    ],

    "capital.gain": [capital_gain],

    "capital.loss": [capital_loss],

    "hours.per.week": [hours_per_week]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict"):

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Income is Greater than 50K")

    else:
        st.error("Income is Less than or Equal to 50K")