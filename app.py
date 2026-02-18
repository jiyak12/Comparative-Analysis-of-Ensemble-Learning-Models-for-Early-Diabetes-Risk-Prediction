import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model & threshold
pipeline = joblib.load("gbm_pipeline.pkl")
optimal_threshold = joblib.load("gbm_threshold.pkl")

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# Sidebar Inputs
st.sidebar.title("📝 Patient Details")

pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 300, 120)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 200, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 0, 100, 30)

# Main Area
st.title("🩺 Diabetes Risk Prediction System")
st.write("Predict diabetes probability using medical parameters.")
st.markdown("---")

if st.button("🔍 Predict Risk"):

    try:
        input_data = pd.DataFrame([{
            "pregnancies": pregnancies,
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetes_pedigree_function": dpf,
            "age": age
        }])

        prob = pipeline.predict_proba(input_data)[0][1]
        percentage = prob * 100

        if prob < optimal_threshold:
            risk_label = "Low Risk"
            color = "green"
        else:
            risk_label = "High Risk"
            color = "red"

        st.subheader("Prediction Result")
        st.progress(int(percentage))

        st.markdown(
            f"""
            <div style='padding:20px;
                        border-radius:10px;
                        background-color:{color};
                        color:white;
                        font-size:20px;
                        text-align:center'>
                {risk_label} ({percentage:.2f}% Probability)
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error occurred: {e}")

st.markdown("---")
st.caption("⚠ Educational use only. Not medical advice.")
