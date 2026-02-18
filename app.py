import streamlit as st
import numpy as np
import joblib
from PIL import Image

# ----------------------------
# Load Saved Model & Threshold
# ----------------------------
pipeline = joblib.load("gbm_pipeline.pkl")
optimal_threshold = joblib.load("gbm_threshold.pkl")

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
}
.risk-box {
    padding:20px;
    border-radius:10px;
    font-size:20px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=120)
    st.markdown("## Enter Patient Details")

    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 300, 120)
    blood_pressure = st.slider("Blood Pressure", 0, 200, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 0, 100, 30)

with col2:
    st.markdown('<p class="big-title">🩺 Diabetes Risk Prediction System</p>', unsafe_allow_html=True)
    st.write("Predict diabetes probability using medical parameters.")
    st.markdown("---")

    if st.button("🔍 Predict Risk"):

        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])

        prob = pipeline.predict_proba(input_data)[0][1]
        percentage = prob * 100

        # Risk Categorization using ROC threshold
        if prob < optimal_threshold * 0.6:
            risk_label = "Low Risk"
            color = "#28a745"
        elif prob < optimal_threshold:
            risk_label = "Moderate Risk"
            color = "#ffc107"
        else:
            risk_label = "High Risk"
            color = "#dc3545"

        st.subheader("Prediction Result")

        # Probability Bar
        st.progress(int(percentage))

        # Risk Box
        st.markdown(
            f"""
            <div class="risk-box" style="background-color:{color}; color:white;">
                {risk_label} ({percentage:.2f}% Probability)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Extra Info
        st.info(f"Optimal Classification Threshold (ROC-based): {optimal_threshold:.3f}")

st.markdown("---")
st.warning("⚠ Disclaimer: This system is for educational purposes only and does not replace medical advice.")
