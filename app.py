import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('salary_model.pkl', 'rb'))

st.title("Employee Salary Predictor")

experience = st.slider("Experience (Years)", 0, 40, 5)
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ['Male', 'Female'])

gender_encoded = 1 if gender == 'Male' else 0

input_df = pd.DataFrame([[experience, age, gender_encoded]], columns=['Experience_Years', 'Age', 'Gender'])

if st.button("Predict Salary"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
