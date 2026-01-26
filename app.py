# =========================================
# SMART FITNESS INSURANCE – STREAMLIT APP
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Smart Fitness Insurance",
    page_icon="💙",
    layout="wide"
)

# -----------------------------------------
# DATA SOURCE (RAW GITHUB)
# -----------------------------------------
DATA_URL = "https://raw.githubusercontent.com/Ramsha121/Smart_Insurance/data/base_plans.csv"

# -----------------------------------------
# LOAD & TRAIN MODEL (SAFE)
# -----------------------------------------
@st.cache_data
def load_and_train():
    df = pd.read_csv(DATA_URL)

    # 🔒 Normalize column names (CRITICAL FIX)
    df.columns = df.columns.str.strip()

    REQUIRED_FEATURES = [
        "Age",
        "Blood Pressure (Systolic)",
        "Blood Pressure (Diastolic)",
        "Heart Beats",
        "BMI",
        "Cholesterol",
        "Steps Taken",
        "Active Minutes",
        "Sleep Duration",
        "Sleep Quality",
        "VO2 Max",
        "Calories Burned",
        "SpO2 Levels",
        "Stress Levels"
    ]

    TARGET = "Claim Amount"

    # ✅ Validate columns
    missing = [c for c in REQUIRED_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error("❌ Dataset column mismatch detected")
        st.write("Missing columns:", missing)
        st.write("Available columns:", list(df.columns))
        st.stop()

    X = df[REQUIRED_FEATURES]
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_scaled, y)

    return df, model, scaler, REQUIRED_FEATURES

df, model, scaler, FEATURES = load_and_train()

# -----------------------------------------
# FITNESS LOGIC
# -----------------------------------------
def fitness_score(u):
    score = (
        0.12 * u["Steps Taken"] +
        0.10 * u["Active Minutes"] +
        0.10 * u["Sleep Duration"] +
        0.08 * u["Sleep Quality"] +
        0.10 * u["VO2 Max"] +
        0.10 * u["SpO2 Levels"] -
        0.15 * u["Stress Levels"] -
        0.08 * u["BMI"]
    )
    return float(np.clip(score, 0, 100))

def category(score):
    if score >= 80: return "Elite"
    elif score >= 65: return "Excellent"
    elif score >= 50: return "Good"
    elif score >= 35: return "Fair"
    else: return "Needs Improvement"

def plan(score):
    if score >= 75: return "Starlite Premium"
    elif score >= 60: return "Starlite Basic"
    elif score >= 45: return "MaxBupa Silver"
    else: return "MaxBupa Gold"

# -----------------------------------------
# HEADER
# -----------------------------------------
st.title("💙 Smart Fitness Insurance")
st.caption("Health-driven personalized insurance intelligence")
st.divider()

# =========================================
# USER INPUT
# =========================================
st.subheader("🧍 Enter Your Health Details")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.slider("Age", 18, 100, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
    sys = st.number_input("BP Systolic", 90, 200, 120)
    dia = st.number_input("BP Diastolic", 60, 130, 80)

with c2:
    heart = st.number_input("Heart Beats", 40, 200, 72)
    chol = st.number_input("Cholesterol", 100, 400, 180)
    spo2 = st.number_input("SpO2 Levels", 85, 100, 98)
    stress = st.slider("Stress Levels", 0, 100, 30)

with c3:
    steps = st.number_input("Steps Taken", 0, 40000, 8000)
    active = st.number_input("Active Minutes", 0, 300, 45)
    sleep = st.number_input("Sleep Duration (hrs)", 0.0, 12.0, 7.0)
    sleep_q = st.slider("Sleep Quality", 0, 100, 70)
    vo2 = st.number_input("VO2 Max", 10.0, 80.0, 40.0)
    calories = st.number_input("Calories Burned", 500, 6000, 2200)

# =========================================
# PREDICTION
# =========================================
if st.button("🔍 Get Personalized Recommendation"):

    user = {
        "Age": age,
        "Blood Pressure (Systolic)": sys,
        "Blood Pressure (Diastolic)": dia,
        "Heart Beats": heart,
        "BMI": bmi,
        "Cholesterol": chol,
        "Steps Taken": steps,
        "Active Minutes": active,
        "Sleep Duration": sleep,
        "Sleep Quality": sleep_q,
        "VO2 Max": vo2,
        "Calories Burned": calories,
        "SpO2 Levels": spo2,
        "Stress Levels": stress
    }

    user_df = pd.DataFrame([user])
    user_scaled = scaler.transform(user_df[FEATURES])

    claim = model.predict(user_scaled)[0]
    score = fitness_score(user)

    st.divider()
    st.subheader("🎯 Personalized Results")

    a, b, c, d = st.columns(4)
    a.metric("Fitness Score", f"{score:.2f}")
    b.metric("Category", category(score))
    c.metric("Estimated Claim", f"₹ {claim:,.0f}")
    d.metric("Recommended Plan", plan(score))

    st.success(
        f"You fall under **{category(score)}** category. "
        f"Recommended plan: **{plan(score)}**."
    )

    # Benchmark plot
    fig = px.scatter(
        df,
        x="Age",
        y="Claim Amount",
        opacity=0.3,
        title="Age vs Claim Amount Benchmark"
    )
    fig.add_scatter(
        x=[age],
        y=[claim],
        mode="markers",
        marker=dict(size=14),
        name="You"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.divider()
st.caption("© 2026 Smart Fitness Insurance | AI-Driven Wellness Protection")
