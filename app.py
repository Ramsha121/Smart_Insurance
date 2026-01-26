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
# HELPERS: DATA CLEANING & SCORING
# -----------------------------------------
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def calculate_enhanced_fitness_score(row):
    """Calculates a normalized health score based on your ML script's logic."""
    # Weighted calculation for health markers
    raw_score = (
        -0.15 * (row.get("Blood Pressure (Systolic)", 120) - 120) / 20 + 
        -0.15 * (row.get("Blood Pressure (Diastolic)", 80) - 80) / 10 +
        -0.10 * (row.get("Heart Beats", 70) - 70) / 30 +
        -0.15 * (row.get("BMI", 22) - 22) / 8 +
        0.20 * row.get("Steps Taken", 0) / 10000 +
        0.15 * row.get("Active Minutes", 0) / 60 +
        0.10 * (row.get("Sleep Duration", 7) - 7) / 2 +
        0.15 * row.get("Sleep Quality", 7) / 10 +
        0.20 * row.get("VO2 Max", 40) / 50 +
        -0.25 * row.get("Stress Levels", 4) / 10
    )
    # Clip and scale for presentation
    return np.clip((raw_score + 1) * 50, 0, 100) 

def categorize_fitness(score):
    """Returns a health category based on the score."""
    if score >= 80: return "🏆 Elite"
    elif score >= 65: return "💪 Excellent"
    elif score >= 50: return "✅ Good"
    elif score >= 35: return "⚡ Fair"
    else: return "🔄 Needs Improvement"

# -----------------------------------------
# DATA LOADING & MODEL TRAINING
# -----------------------------------------
@st.cache_data
def load_data():
    base = clean_columns(pd.read_csv(BASE_PLANS_URL))
    fitness = pd.read_csv(FITNESS_URL)
    fitness.columns = fitness.columns.str.strip()
    return base, fitness

base_df, fitness_df = load_data()

FITNESS_FEATURES = [
    "Age", "Blood Pressure (Systolic)", "Blood Pressure (Diastolic)",
    "Heart Beats", "BMI", "Cholesterol", "Steps Taken", "Active Minutes",
    "Sleep Duration", "Sleep Quality", "VO2 Max", "Calories Burned",
    "SpO2 Levels", "Stress Levels"
]

@st.cache_resource
def train_model():
    X = fitness_df[FITNESS_FEATURES]
    y = fitness_df["Claim Amount"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    model.fit(scaler.fit_transform(X), y)
    return model, scaler

model, scaler = train_model()

# -----------------------------------------
# UI - INPUT SECTION
# -----------------------------------------
st.title("💙 Smart Fitness-Based Insurance System")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    name = st.text_input("Full Name", "User")
    age = st.slider("Age", 18, 80, 30)
    occupation = st.selectbox("Occupation", sorted(base_df["occupation"].unique()))

with col2:
    income = st.number_input("Annual Income (₹)", 50000, 5000000, 500000)
    bmi = st.number_input("BMI", 15.0, 45.0, 22.5)
    stress = st.slider("Stress Level (1-10)", 1, 10, 4)

with col3:
    steps = st.number_input("Average Daily Steps", 0, 30000, 8000)
    sleep = st.slider("Sleep Quality (1-10)", 1, 10, 7)
    vo2 = st.slider("VO2 Max", 20.0, 60.0, 40.0)

# -----------------------------------------
# UI - OUTPUT SECTION
# -----------------------------------------
if st.button("🔍 Generate Personalized Health & Policy Report"):
    # Prepare User Data
    user_data = {
        "Age": age, "Blood Pressure (Systolic)": 120, "Blood Pressure (Diastolic)": 80,
        "Heart Beats": 72, "BMI": bmi, "Cholesterol": 190, "Steps Taken": steps,
        "Active Minutes": 45, "Sleep Duration": 7.5, "Sleep Quality": sleep,
        "VO2 Max": vo2, "Calories Burned": 2200, "SpO2 Levels": 98, "Stress Levels": stress
    }
    
    # Calculate Results
    score = calculate_enhanced_fitness_score(user_data)
    category = categorize_fitness(score)
    prediction = model.predict(scaler.transform(pd.DataFrame([user_data])))[0]
    
    # Display Metrics
    st.divider()
    st.subheader(f"🎯 Analysis for {name}")
    m1, m2, m3 = st.columns(3)
    m1.metric("🏃 Health Score", f"{score:.1f}/100")
    m2.metric("📋 Health Status", category)
    m3.metric("💰 Risk-Based Premium Estimate", f"₹{prediction:,.0f}")

    # 3D Visualisation
    st.subheader("🌟 3D Health Universe")
    st.caption("Visualizing your fitness status relative to population trends")
    
    # Add user to a sample of the population for the graph
    plot_df = fitness_df.sample(200).copy()
    user_plot_point = pd.DataFrame([user_data])
    user_plot_point["Name"] = "YOU"
    user_plot_point["Status"] = "Target"
    
    fig_3d = px.scatter_3d(
        plot_df, x='Age', y='BMI', z='Steps Taken',
        color='Stress Levels', size='Heart Beats',
        color_continuous_scale='Viridis',
        title='Multi-dimensional Wellness Mapping'
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Business Insight Recommendation
    st.info(f"💡 **Recommendation:** Based on your **{category}** fitness level, you qualify for high-tier wellness discounts.")
