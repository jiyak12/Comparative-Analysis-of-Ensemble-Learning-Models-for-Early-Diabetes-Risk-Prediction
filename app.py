import streamlit as st
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Page background */
    .stApp { background-color: #f7f8fc; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8eaf0;
        padding-top: 2rem;
    }
    section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

    /* Sidebar label */
    .sidebar-title {
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 1.4rem;
    }

    /* Slider tweaks */
    .stSlider > div > div > div { background: #e0e7ff !important; }
    .stSlider > div > div > div > div { background: #4f46e5 !important; }

    /* Main card */
    .main-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2.5rem 2.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 20px rgba(0,0,0,0.04);
        max-width: 680px;
        margin: 3rem auto 0 auto;
    }

    /* Hero */
    .hero-icon { font-size: 2.4rem; margin-bottom: 0.3rem; }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: #111827;
        margin: 0 0 0.35rem 0;
        line-height: 1.25;
    }
    .hero-sub {
        font-size: 0.93rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid #f0f1f5;
        margin: 1.8rem 0;
    }

    /* Predict button */
    div[data-testid="stButton"] > button {
        background: #4f46e5;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-size: 0.93rem;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.2s ease, transform 0.1s ease;
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        background: #4338ca;
        transform: translateY(-1px);
    }
    div[data-testid="stButton"] > button:active { transform: translateY(0); }

    /* Result cards */
    .result-wrapper { margin-top: 1.6rem; }
    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 0.6rem;
    }

    .risk-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        border-radius: 9999px;
        padding: 0.45rem 1.1rem;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 0.9rem;
    }
    .risk-low  { background:#ecfdf5; color:#065f46; }
    .risk-mid  { background:#fffbeb; color:#92400e; }
    .risk-high { background:#fef2f2; color:#991b1b; }

    /* Progress bar override */
    .stProgress > div > div > div > div { border-radius: 9999px; }
    .stProgress > div > div { border-radius: 9999px; background: #f0f1f5; }

    /* Probability big number */
    .prob-number {
        font-size: 3rem;
        font-weight: 700;
        color: #111827;
        line-height: 1;
        margin-bottom: 0.1rem;
    }
    .prob-unit { font-size: 1rem; color: #6b7280; font-weight: 400; }

    /* Disclaimer */
    .disclaimer {
        margin-top: 2rem;
        font-size: 0.78rem;
        color: #9ca3af;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">Patient Parameters</p>', unsafe_allow_html=True)

    pregnancies     = st.slider("Pregnancies",                0,    20,   1)
    glucose         = st.slider("Glucose Level (mg/dL)",      0,   200, 100)
    blood_pressure  = st.slider("Blood Pressure (mmHg)",      0,   150,  70)
    skin_thickness  = st.slider("Skin Thickness (mm)",        0,   100,  20)
    insulin         = st.slider("Insulin (μU/mL)",            0,   900,  80)
    bmi             = st.slider("BMI",                        0.0, 70.0, 25.0, step=0.1)
    dpf             = st.slider("Diabetes Pedigree Function", 0.0,  3.0,  0.5, step=0.01)
    age             = st.slider("Age",                        1,   120,  30)

# ── Main card ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Hero
st.markdown('<div class="hero-icon">🩺</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Enter patient parameters in the sidebar, then run the assessment.</p>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Predict button
predict = st.button("Run Risk Assessment")

if predict:
    input_data  = np.array([[pregnancies, glucose, blood_pressure,
                              skin_thickness, insulin, bmi, dpf, age]])
    probability = model.predict_proba(input_data)[0][1]
    pct         = probability * 100

    st.markdown('<div class="result-wrapper">', unsafe_allow_html=True)
    st.markdown('<p class="result-label">Assessment Result</p>', unsafe_allow_html=True)

    # Big probability number
    st.markdown(
        f'<p class="prob-number">{pct:.1f}<span class="prob-unit"> %</span></p>'
        f'<p style="color:#6b7280;font-size:0.85rem;margin:0 0 1rem 0;">Estimated diabetes probability</p>',
        unsafe_allow_html=True,
    )

    # Progress bar
    st.progress(int(pct))

    # Risk chip
    if pct < 30:
        chip = f'<span class="risk-chip risk-low">● Low Risk</span>'
    elif pct < 70:
        chip = f'<span class="risk-chip risk-mid">● Moderate Risk</span>'
    else:
        chip = f'<span class="risk-chip risk-high">● High Risk</span>'

    st.markdown(chip, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p class="disclaimer">⚠️ For educational purposes only — not a substitute for professional medical advice.</p>',
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)
