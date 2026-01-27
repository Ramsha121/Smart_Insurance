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
# DATA SOURCE (GitHub RAW)
# -----------------------------------------
DATA_URL = "https://raw.githubusercontent.com/Ramsha121/Smart_Insurance/data/base_plans.csv"

# -----------------------------------------
# LOAD & TRAIN MODEL
# -----------------------------------------
@st.cache_data
def load_and_train():
    df = pd.read_csv(DATA_URL)

    target = "Claim Amount"

    features = [
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

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_scaled, y)

    return df, model, scaler, features

df, model, scaler, FEATURES = load_and_train()

# -----------------------------------------
# FITNESS SCORE LOGIC
# -----------------------------------------
def compute_fitness_score(user):
    score = (
        0.12 * user["Steps Taken"] +
        0.10 * user["Active Minutes"] +
        0.10 * user["Sleep Duration"] +
        0.08 * user["Sleep Quality"] +
        0.10 * user["VO2 Max"] +
        0.10 * user["SpO2 Levels"] -
        0.15 * user["Stress Levels"] -
        0.08 * user["BMI"]
    )
    return np.clip(score, 0, 100)

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
st.markdown("## 💙 Smart Fitness Insurance")
st.caption("Personalized health-driven insurance recommendations")
st.divider()

# =========================================
# 1️⃣ USER INPUT SECTION
# =========================================
st.subheader("🧍 Enter Your Health & Fitness Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
    systolic = st.number_input("BP Systolic", 90, 200, 120)
    diastolic = st.number_input("BP Diastolic", 60, 130, 80)

with col2:
    heart = st.number_input("Heart Beats", 40, 200, 72)
    cholesterol = st.number_input("Cholesterol", 100, 400, 180)
    spo2 = st.number_input("SpO2 Levels", 85, 100, 98)
    stress = st.slider("Stress Levels", 0, 100, 30)

with col3:
    steps = st.number_input("Steps Taken", 0, 40000, 8000)
    active = st.number_input("Active Minutes", 0, 300, 45)
    sleep = st.number_input("Sleep Duration (hrs)", 0.0, 12.0, 7.0)
    sleep_q = st.slider("Sleep Quality", 0, 100, 70)
    vo2 = st.number_input("VO2 Max", 10.0, 80.0, 40.0)
    calories = st.number_input("Calories Burned", 500, 6000, 2200)

# =========================================
# 2️⃣ PERSONALIZED OUTPUT
# =========================================
if st.button("🔍 Get My Insurance Recommendation"):

    user_input = {
        "Age": age,
        "Blood Pressure (Systolic)": systolic,
        "Blood Pressure (Diastolic)": diastolic,
        "Heart Beats": heart,
        "BMI": bmi,
        "Cholesterol": cholesterol,
        "Steps Taken": steps,
        "Active Minutes": active,
        "Sleep Duration": sleep,
        "Sleep Quality": sleep_q,
        "VO2 Max": vo2,
        "Calories Burned": calories,
        "SpO2 Levels": spo2,
        "Stress Levels": stress
    }

    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df[FEATURES])

    predicted_claim = model.predict(user_scaled)[0]
    fitness_score = compute_fitness_score(user_input)

    st.divider()
    st.subheader("🎯 Personalized Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fitness Score", f"{fitness_score:.2f}")
    c2.metric("Health Category", category(fitness_score))
    c3.metric("Estimated Claim Amount", f"₹ {predicted_claim:,.0f}")
    c4.metric("Recommended Plan", plan(fitness_score))

    st.success(
        f"You fall under **{category(fitness_score)}** category. "
        f"Best suited plan: **{plan(fitness_score)}**."
    )

    # -----------------------------------------
    # VISUAL COMPARISON
    # -----------------------------------------
    st.subheader("📊 Where You Stand")

    df_viz = df.copy()
    df_viz["User"] = "Others"

    user_viz = user_df.copy()
    user_viz["Fitness Score"] = fitness_score
    user_viz["User"] = "You"

    fig = px.scatter(
        df_viz,
        x="Age",
        y="Claim Amount",
        opacity=0.4,
        title="Age vs Claim Amount (Benchmarking)"
    )
    fig.add_scatter(
        x=[age],
        y=[predicted_claim],
        mode="markers",
        marker=dict(size=14),
        name="You"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.divider()
st.caption("© 2026 Smart Fitness Insurance | Data-Driven Wellness & Protection")
