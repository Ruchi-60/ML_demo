import streamlit as st
import joblib
import numpy as np

model = joblib.load("student_model.pkl")

st.title("Student Pass/Fail Predictor")

st.write("Enter student details")

study_hours = st.number_input("Study Hours",0,12)


if st.button("Predict"):
    input_data= np.array([[study_hours]])
    prediction = model.predict(input_data)
    
    if prediction[0] >= 1:
        st.success("Student will PASS")
        st.write("score: ",prediction[0])
    else:
        st.error("Student will FAIL")
