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
    page_title="Smart Insurance Intelligence",
    page_icon="💙",
    layout="wide"
)

# -----------------------------------------
# DATA URLs (RAW GITHUB)
# -----------------------------------------
BASE_PLANS_URL = "https://raw.githubusercontent.com/Ramsha121/Smart_Insurance/data/base_plans.csv"
FITNESS_URL = "https://raw.githubusercontent.com/Ramsha121/Smart_Insurance/data/fitness_claim_dataset.csv"

# -----------------------------------------
# HELPER: REMOVE DUPLICATE COLUMNS
# -----------------------------------------
def clean_columns(df):
    """Strips whitespace, lowers case, and removes duplicate column names."""
    df.columns = df.columns.str.strip().str.lower()
    # Identifies duplicates and appends a suffix (e.g., 'premium', 'premium.1')
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data
def load_base_plans():
    df = pd.read_csv(BASE_PLANS_URL)
    df = clean_columns(df)
    return df

@st.cache_data
def load_fitness_data():
    df = pd.read_csv(FITNESS_URL)
    # Fitness data usually has specific casing, let's keep it but strip whitespace
    df.columns = df.columns.str.strip()
    # Check for duplicates even here to be safe
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df

base_df = load_base_plans()
fitness_df = load_fitness_data()

# -----------------------------------------
# TRAIN FITNESS → CLAIM MODEL
# -----------------------------------------
FITNESS_FEATURES = [
    "Age", "Blood Pressure (Systolic)", "Blood Pressure (Diastolic)",
    "Heart Beats", "BMI", "Cholesterol", "Steps Taken", "Active Minutes",
    "Sleep Duration", "Sleep Quality", "VO2 Max", "Calories Burned",
    "SpO2 Levels", "Stress Levels"
]
TARGET = "Claim Amount"

@st.cache_resource
def train_claim_model():
    # Filter only features present in the DF to avoid KeyError
    available_features = [f for f in FITNESS_FEATURES if f in fitness_df.columns]
    X = fitness_df[available_features]
    y = fitness_df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, available_features

claim_model, scaler, final_features = train_claim_model()

# -----------------------------------------
# HELPERS
# -----------------------------------------
def fitness_score(row):
    score = (
        0.15 * row.get("Steps Taken", 0)
        + 0.15 * row.get("Active Minutes", 0)
        + 0.1 * row.get("Sleep Duration", 0)
        + 0.1 * row.get("Sleep Quality", 0)
        + 0.1 * row.get("VO2 Max", 0)
        - 0.15 * row.get("Stress Levels", 0)
        - 0.1 * row.get("BMI", 0)
    )
    return np.clip(score, 0, 100)

def recommend_policy(occupation):
    # Ensure column name matches the cleaned 'occupation'
    df = base_df[base_df["occupation"].str.lower() == occupation.lower()]
    if df.empty:
        return "Starlite Basic"
    # Logic: Find policy with lowest premium for this occupation
    return df.groupby("insurance")["premium"].mean().idxmin()

# -----------------------------------------
# UI HEADER
# -----------------------------------------
st.title("💙 Smart Fitness-Based Insurance System")
st.divider()

# USER INPUTS
col1, col2, col3 = st.columns(3)
with col1:
    name = st.text_input("Name", "User")
    age = st.slider("Age", 18, 80, 30)
    occ_list = sorted(base_df["occupation"].unique()) if not base_df.empty else ["Service"]
    occupation = st.selectbox("Occupation", occ_list)

with col2:
    income = st.number_input("Annual Income (₹)", 50000, 5000000, 400000)
    bmi = st.number_input("BMI", 15.0, 40.0, 22.0)
    stress = st.slider("Stress Level", 1, 10, 4)

with col3:
    steps = st.number_input("Daily Steps", 1000, 20000, 8000)
    sleep = st.slider("Sleep Duration (hrs)", 4.0, 10.0, 7.0)
    vo2 = st.slider("VO2 Max", 20.0, 60.0, 40.0)

if st.button("🔍 Generate Personalized Recommendation"):
    user_row = {
        "Age": age, "Blood Pressure (Systolic)": 120, "Blood Pressure (Diastolic)": 80,
        "Heart Beats": 72, "BMI": bmi, "Cholesterol": 180, "Steps Taken": steps,
        "Active Minutes": 45, "Sleep Duration": sleep, "Sleep Quality": 7,
        "VO2 Max": vo2, "Calories Burned": 2200, "SpO2 Levels": 98, "Stress Levels": stress
    }

    user_df = pd.DataFrame([user_row])
    user_scaled = scaler.transform(user_df[final_features])
    predicted_claim = claim_model.predict(user_scaled)[0]
    score = fitness_score(user_row)
    policy = recommend_policy(occupation)

    st.divider()
    st.subheader(f"🎯 Personalized Results for {name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("🏃 Fitness Score", f"{score:.1f}")
    c2.metric("💰 Estimated Claim", f"₹{predicted_claim:,.0f}")
    c3.metric("📄 Recommended Policy", policy)

    # VISUALIZATIONS
    st.subheader("📊 Insurance Policy Distribution")
    # This part now works because base_df columns are unique
    fig1 = px.pie(base_df, names="insurance", title="Overall Policy Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    st.info("💡 **Insight:** Your recommendation is based on lifestyle markers and regional policy data.")

st.divider()
st.caption("© 2026 Smart Insurance | Health meets Intelligence")
