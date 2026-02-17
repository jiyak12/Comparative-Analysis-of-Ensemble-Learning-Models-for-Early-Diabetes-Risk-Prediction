import streamlit as st
import joblib
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# ── CSS — minimal, safe overrides only ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #f4f6fb; }

#MainMenu, footer { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

div[data-testid="stButton"] > button {
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.8rem;
    font-weight: 500;
    font-size: 0.95rem;
    width: 100%;
    transition: background 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background: #4338ca;
    color: white;
}

div[data-testid="stProgress"] > div { border-radius: 99px; }
div[data-testid="stProgress"] > div > div { border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Patient Parameters")
    st.markdown("---")
    pregnancies    = st.slider("Pregnancies",                0,    20,   1)
    glucose        = st.slider("Glucose Level (mg/dL)",      0,   200, 100)
    blood_pressure = st.slider("Blood Pressure (mmHg)",      0,   150,  70)
    skin_thickness = st.slider("Skin Thickness (mm)",        0,   100,  20)
    insulin        = st.slider("Insulin (μU/mL)",            0,   900,  80)
    bmi            = st.slider("BMI",                        0.0, 70.0, 25.0, step=0.1)
    dpf            = st.slider("Diabetes Pedigree Function", 0.0,  3.0,  0.5, step=0.01)
    age            = st.slider("Age",                        1,   120,  30)

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# 🩺 Diabetes Risk Predictor")
st.markdown("Adjust the patient parameters in the sidebar, then click **Run Assessment**.")
st.markdown("---")

if st.button("Run Assessment"):
    input_data  = np.array([[pregnancies, glucose, blood_pressure,
                              skin_thickness, insulin, bmi, dpf, age]])
    probability = model.predict_proba(input_data)[0][1]
    pct         = probability * 100

    st.markdown("#### Result")

    # Big probability number
    st.markdown(f"## {pct:.1f}%")
    st.caption("Estimated probability of diabetes")

    # Progress bar
    st.progress(int(pct))
    st.markdown("")

    # Risk level using native Streamlit components (reliable rendering)
    if pct < 30:
        st.success(f"🟢 Low Risk — {pct:.1f}%")
    elif pct < 70:
        st.warning(f"🟡 Moderate Risk — {pct:.1f}%")
    else:
        st.error(f"🔴 High Risk — {pct:.1f}%")

    # Input summary collapsible
    with st.expander("View input summary"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Glucose", glucose)
            st.metric("BMI", bmi)
            st.metric("Age", age)
            st.metric("Pregnancies", pregnancies)
        with col2:
            st.metric("Blood Pressure", blood_pressure)
            st.metric("Skin Thickness", skin_thickness)
            st.metric("Insulin", insulin)
            st.metric("DPF", dpf)

st.markdown("---")
st.caption("⚠️ For educational purposes only — not a substitute for professional medical advice.")
