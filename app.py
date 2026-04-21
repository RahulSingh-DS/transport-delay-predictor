import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# ABSOLUTE PATH FIX
# Finds model/ folder relative to app.py itself,
# no matter where you run streamlit from
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check both root/model/ AND notebook/model/ automatically
for candidate in [
    os.path.join(BASE_DIR, "model"),
    os.path.join(BASE_DIR, "notebook", "model"),
]:
    if os.path.exists(os.path.join(candidate, "lgbm_model.pkl")):
        MODEL_DIR = candidate
        break
else:
    MODEL_DIR = os.path.join(BASE_DIR, "model")  # fallback for error message

MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "label_encoders.pkl")
FEAT_PATH  = os.path.join(MODEL_DIR, "feature_columns.pkl")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Transit Delay Predictor",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }
    .hero-card {
        background: linear-gradient(135deg, #1f6feb22, #388bfd11);
        border: 1px solid #30363d; border-radius: 16px;
        padding: 2rem; margin-bottom: 1.5rem; text-align: center;
    }
    .hero-title {
        font-family: 'Space Mono', monospace; font-size: 2.4rem;
        font-weight: 700; color: #58a6ff; margin: 0;
    }
    .hero-sub { color: #8b949e; font-size: 1rem; margin-top: 0.5rem; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-label { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #f0f6fc; }
    .result-box-green  { background: #0d4429; border: 1px solid #26a641; border-radius: 12px; padding: 1.5rem; text-align: center; }
    .result-box-yellow { background: #2d1e00; border: 1px solid #d29922; border-radius: 12px; padding: 1.5rem; text-align: center; }
    .result-box-red    { background: #3d1a1a; border: 1px solid #f85149; border-radius: 12px; padding: 1.5rem; text-align: center; }
    .result-minutes { font-family: 'Space Mono', monospace; font-size: 3rem; font-weight: 700; }
    .result-label { color: #8b949e; font-size: 0.9rem; margin-top: 0.3rem; }
    .sidebar-header {
        font-family: 'Space Mono', monospace; color: #58a6ff; font-size: 0.85rem;
        letter-spacing: 1.5px; text-transform: uppercase;
        margin-bottom: 0.5rem; border-bottom: 1px solid #30363d; padding-bottom: 0.4rem;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label { color: #c9d1d9 !important; font-size: 0.88rem; }
    .stButton > button {
        background: linear-gradient(90deg, #1f6feb, #388bfd); color: white;
        border: none; border-radius: 8px; font-family: 'Space Mono', monospace;
        font-size: 0.95rem; padding: 0.6rem 1.5rem; width: 100%; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .about-box {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 1.2rem 1.5rem; color: #8b949e; font-size: 0.88rem; line-height: 1.7;
    }
    .info-box {
        background: #0d2137; border: 1px solid #1f6feb; border-radius: 10px;
        padding: 1rem 1.2rem; color: #79c0ff; font-size: 0.88rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    m  = joblib.load(MODEL_PATH)
    e  = joblib.load(ENC_PATH)
    fc = joblib.load(FEAT_PATH)
    return m, e, fc

model_loaded = (
    os.path.exists(MODEL_PATH) and
    os.path.exists(ENC_PATH)   and
    os.path.exists(FEAT_PATH)
)

if model_loaded:
    model, label_encoders, feature_columns = load_artifacts()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <p class="hero-title">🚌 Transit Delay Predictor</p>
    <p class="hero-sub">Predict public transport delays using real-world weather, events & timetable data</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"""
    ❌ Model files not found. Make sure these 3 files exist:
    - {MODEL_PATH}
    - {ENC_PATH}
    - {FEAT_PATH}

    Run your Jupyter notebook to generate them, then restart the app.
    """)
    st.stop()

# Show where model was found
st.markdown(f'<div class="info-box">✅ Model loaded from: <code>{MODEL_DIR}</code></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-header">🗓 Schedule Info</p>', unsafe_allow_html=True)
    hour         = st.slider("Departure Hour (0–23)", 0, 23, 8)
    day_of_week  = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    month        = st.selectbox("Month", list(range(1, 13)), index=0)
    vehicle_type = st.selectbox("Vehicle Type", ["Bus", "Train"])

    st.markdown('<p class="sidebar-header">🌦 Weather Conditions</p>', unsafe_allow_html=True)
    weather     = st.selectbox("Weather", ["Clear", "Rain", "Fog", "Snow", "Windy"])
    temperature = st.slider("Temperature (°C)", -10, 45, 22)
    wind_speed  = st.slider("Wind Speed (km/h)", 0, 100, 15)
    visibility  = st.slider("Visibility (km)", 0, 20, 10)

    st.markdown('<p class="sidebar-header">📍 Route & Events</p>', unsafe_allow_html=True)
    is_event    = st.selectbox("Event Nearby?", ["No", "Yes"])
    route_type  = st.selectbox("Route Type", ["Urban", "Suburban", "Rural"])

    predict_btn = st.button("⚡ Predict Delay")

# ─────────────────────────────────────────────
# DERIVED FEATURES
# ─────────────────────────────────────────────
day_map      = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
is_weekend   = 1 if day_map[day_of_week] >= 5 else 0
is_peak_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
is_event_bin = 1 if is_event == "Yes" else 0

# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Departure Hour</div>
        <div class="metric-value">{hour:02d}:00</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Weather</div>
        <div class="metric-value">{weather}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Peak Hour</div>
        <div class="metric-value">{'Yes ⚠️' if is_peak_hour else 'No ✅'}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if predict_btn:
    input_dict = {
        "hour": hour,
        "day_of_week": day_map[day_of_week],
        "month": month,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "temperature": temperature,
        "wind_speed": wind_speed,
        "visibility_km": visibility,
        "is_event_nearby": is_event_bin,
        "weather_condition": weather,
        "vehicle_type": vehicle_type,
        "route_type": route_type,
    }
    input_df = pd.DataFrame([input_dict])

    for col, le in label_encoders.items():
        if col in input_df.columns:
            val = input_df[col].astype(str).values[0]
            input_df[col] = le.transform([val]) if val in le.classes_ else 0

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    prediction = max(0, round(float(model.predict(input_df)[0]), 1))

    if prediction < 5:
        box_class, status, color = "result-box-green",  "✅ On Time",     "#3fb950"
    elif prediction < 15:
        box_class, status, color = "result-box-yellow", "⚠️ Minor Delay", "#d29922"
    else:
        box_class, status, color = "result-box-red",    "🚨 Major Delay", "#f85149"

    st.markdown(f"""
    <div class="{box_class}">
        <div class="result-minutes" style="color:{color}">{prediction} min</div>
        <div class="result-label">{status} — Predicted Delay</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 Top Contributing Features"):
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)
        st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)

# ─────────────────────────────────────────────
# ABOUT
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
    <b style="color:#c9d1d9">About this Project</b><br><br>
    This app uses a <b>LightGBM regression model</b> trained on the
    <i>Public Transport Delays with Weather & Events</i> dataset from Kaggle.
    The model learns from timetable data, weather conditions (rain, fog, temperature, wind),
    and local event flags to predict expected delay in minutes for buses and trains.<br><br>
    <b>Model:</b> LightGBM Regressor &nbsp;|&nbsp;
    <b>Metrics:</b> MAE, RMSE, R² &nbsp;|&nbsp;
    <b>Features:</b> 12+ engineered features
</div>
""", unsafe_allow_html=True)
