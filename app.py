# =========================================
# SMART FITNESS + INSURANCE RECOMMENDATION
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Smart Insurance Intelligence",
    page_icon="💙",
    layout="wide"
)

# -----------------------------------------
# DATA URLs (RAW GITHUB)
# -----------------------------------------
BASE_PLANS_URL = "https://github.com/Ramsha121/Smart_Insurance/blob/data/base_plans.csv"
FITNESS_URL = "https://raw.githubusercontent.com/Ramsha121/Smart_Insurance/data/fitness_claims.csv"

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data
def load_base_plans():
    df = pd.read_csv(BASE_PLANS_URL)
    df.columns = df.columns.str.strip().str.lower()
    return df

@st.cache_data
def load_fitness_data():
    df = pd.read_csv(FITNESS_URL)
    df.columns = df.columns.str.strip()
    return df

base_df = load_base_plans()
fitness_df = load_fitness_data()

# -----------------------------------------
# TRAIN FITNESS → CLAIM MODEL
# -----------------------------------------
FITNESS_FEATURES = [
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

@st.cache_resource
def train_claim_model():
    X = fitness_df[FITNESS_FEATURES]
    y = fitness_df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_scaled, y)

    return model, scaler

claim_model, scaler = train_claim_model()

# -----------------------------------------
# HELPERS
# -----------------------------------------
def fitness_score(row):
    score = (
        0.15 * row["Steps Taken"]
        + 0.15 * row["Active Minutes"]
        + 0.1 * row["Sleep Duration"]
        + 0.1 * row["Sleep Quality"]
        + 0.1 * row["VO2 Max"]
        - 0.15 * row["Stress Levels"]
        - 0.1 * row["BMI"]
    )
    return np.clip(score, 0, 100)

def recommend_policy(age, income, occupation):
    df = base_df[base_df["occupation"].str.lower() == occupation.lower()]
    if df.empty:
        return "Starlite Basic"
    return df.groupby("insurance")["premium"].mean().idxmin()

# -----------------------------------------
# HEADER
# -----------------------------------------
st.title("💙 Smart Fitness-Based Insurance System")
st.caption("Personalized health insights + intelligent policy recommendation")
st.divider()

# =========================================
# USER INPUT SECTION
# =========================================
st.subheader("🧍 Enter Your Details")

col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Name")
    age = st.slider("Age", 18, 80, 30)
    occupation = st.selectbox(
        "Occupation",
        sorted(base_df["occupation"].unique())
    )

with col2:
    income = st.number_input("Annual Income (₹)", 50000, 5000000, 400000)
    bmi = st.number_input("BMI", 15.0, 40.0, 22.0)
    stress = st.slider("Stress Level", 1, 10, 4)

with col3:
    steps = st.number_input("Daily Steps", 1000, 20000, 8000)
    sleep = st.slider("Sleep Duration (hrs)", 4.0, 10.0, 7.0)
    vo2 = st.slider("VO2 Max", 20.0, 60.0, 40.0)

# -----------------------------------------
# PREDICTION
# -----------------------------------------
if st.button("🔍 Generate Personalized Recommendation"):

    user_row = {
        "Age": age,
        "Blood Pressure (Systolic)": 120,
        "Blood Pressure (Diastolic)": 80,
        "Heart Beats": 72,
        "BMI": bmi,
        "Cholesterol": 180,
        "Steps Taken": steps,
        "Active Minutes": 45,
        "Sleep Duration": sleep,
        "Sleep Quality": 7,
        "VO2 Max": vo2,
        "Calories Burned": 2200,
        "SpO2 Levels": 98,
        "Stress Levels": stress
    }

    user_df = pd.DataFrame([user_row])
    user_scaled = scaler.transform(user_df[FITNESS_FEATURES])

    predicted_claim = claim_model.predict(user_scaled)[0]
    score = fitness_score(user_row)
    policy = recommend_policy(age, income, occupation)

    # -----------------------------------------
    # OUTPUT
    # -----------------------------------------
    st.divider()
    st.subheader(f"🎯 Personalized Results for {name}")

    c1, c2, c3 = st.columns(3)
    c1.metric("🏃 Fitness Score", f"{score:.1f}")
    c2.metric("💰 Estimated Claim", f"₹{predicted_claim:,.0f}")
    c3.metric("📄 Recommended Policy", policy)

    st.success(f"✅ Best policy for you: **{policy}**")

    # -----------------------------------------
    # VISUALIZATION 1: POLICY DISTRIBUTION
    # -----------------------------------------
    st.subheader("📊 Insurance Policy Distribution")
    fig1 = px.pie(
        base_df,
        names="insurance",
        title="Overall Policy Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------------------
    # VISUALIZATION 2: OCCUPATION vs POLICY
    # -----------------------------------------
    st.subheader("📊 Occupation–Policy Preference")
    occ_matrix = pd.crosstab(
        base_df["occupation"],
        base_df["insurance"]
    )
    fig2 = px.imshow(
        occ_matrix,
        text_auto=True,
        aspect="auto"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------
    # BUSINESS INSIGHTS
    # -----------------------------------------
    st.subheader("🧠 Business Insights")

    insight_df = base_df.groupby("insurance").agg(
        Customers=("insurance", "count"),
        AvgPremium=("premium", "mean"),
        TotalRevenue=("premium", "sum")
    ).reset_index()

    insight_df["Rev/Customer"] = (
        insight_df["TotalRevenue"] / insight_df["Customers"]
    )

    st.dataframe(insight_df, use_container_width=True)

    st.info(
        "💡 **Insight:** StarLite plans dominate across occupations with strong revenue stability, "
        "while MaxBupa plans cater to niche high-risk segments."
    )

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.divider()
st.caption("© 2026 Smart Insurance | Health meets Intelligence")
