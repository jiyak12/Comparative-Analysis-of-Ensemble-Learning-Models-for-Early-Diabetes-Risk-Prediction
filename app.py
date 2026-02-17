import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# Load model
model = joblib.load("diabetes_model.pkl")

# Header
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("Predict the probability of diabetes using medical parameters.")

st.markdown("---")

# Sidebar inputs
st.sidebar.header("Enter Patient Details")

pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 100)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 150, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.slider("Age", 1, 120, 30)

# Prediction button
if st.button("🔍 Predict Risk"):

    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("## 📊 Prediction Result")

    # Progress bar
    st.progress(int(probability * 100))

    # Risk interpretation
    if probability < 0.30:
        st.success(f"🟢 Low Risk ({probability*100:.2f}%)")
    elif probability < 0.70:
        st.warning(f"🟡 Moderate Risk ({probability*100:.2f}%)")
    else:
        st.error(f"🔴 High Risk ({probability*100:.2f}%)")

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: This application is for educational purposes only "
           "and should not replace professional medical consultation.")
