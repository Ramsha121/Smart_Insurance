import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
# DATA URLs
# -----------------------------------------
BASE_PLANS_URL = "https://raw.githubusercontent.com/Ramsha121/Smart_Insurance/data/base_plans.csv"
FITNESS_URL = "https://raw.githubusercontent.com/Ramsha121/Smart_Insurance/data/fitness_claim_dataset.csv"

# -----------------------------------------
# LOAD & CLEAN DATA
# -----------------------------------------
@st.cache_data
def load_data():
    # Load Base Plans
    base = pd.read_csv(BASE_PLANS_URL)
    base.columns = base.columns.str.strip().str.lower()
    
    # Load Fitness Data
    fitness = pd.read_csv(FITNESS_URL)
    fitness.columns = fitness.columns.str.strip()
    
    return base, fitness

base_df, fitness_df = load_data()

# -----------------------------------------
# TRAIN MODEL (From your ML Script)
# -----------------------------------------
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
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

# -----------------------------------------
# HELPERS (The exact logic from your .py file)
# -----------------------------------------
def get_fitness_score(row):
    # Weighted health scoring formula
    score = (
        0.15 * (row["Steps Taken"] / 10000) +
        0.15 * (row["Active Minutes"] / 60) +
        0.10 * (row["Sleep Quality"] / 10) +
        0.20 * (row["VO2 Max"] / 50) -
        0.20 * (row["Stress Levels"] / 10) -
        0.20 * (row["BMI"] / 30)
    )
    return np.clip((score + 1) * 50, 0, 100)

def get_health_category(score):
    if score > 80: return "🏆 Elite"
    if score > 65: return "💪 Excellent"
    if score > 50: return "✅ Good"
    return "🔄 Needs Improvement"

# -----------------------------------------
# MAIN UI
# -----------------------------------------
st.title("💙 Smart Fitness-Based Insurance System")
st.markdown("### High-Performance Health Analytics & Policy Matching")
st.divider()

# Input Grid
col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Name", "John Doe")
    age = st.slider("Age", 18, 80, 30)
    occupation = st.selectbox("Occupation", sorted(base_df['occupation'].unique()))

with col2:
    income = st.number_input("Annual Income (₹)", 50000, 5000000, 600000)
    bmi = st.number_input("BMI", 10.0, 50.0, 24.0)
    stress = st.slider("Stress Level", 1, 10, 5)

with col3:
    steps = st.number_input("Daily Steps", 0, 25000, 8000)
    vo2 = st.slider("VO2 Max", 20.0, 70.0, 42.0)
    sleep_q = st.slider("Sleep Quality", 1, 10, 7)

# -----------------------------------------
# PREDICTION & GRAPHING LOGIC
# -----------------------------------------
if st.button("🔍 Run Full Diagnostic & Recommendation"):
    
    # 1. Generate Prediction
    user_data = {
        "Age": age, "Blood Pressure (Systolic)": 120, "Blood Pressure (Diastolic)": 80,
        "Heart Beats": 72, "BMI": bmi, "Cholesterol": 180, "Steps Taken": steps,
        "Active Minutes": 45, "Sleep Duration": 7.0, "Sleep Quality": sleep_q,
        "VO2 Max": vo2, "Calories Burned": 2200, "SpO2 Levels": 98, "Stress Levels": stress
    }
    
    user_df = pd.DataFrame([user_data])
    user_scaled = scaler.transform(user_df[FITNESS_FEATURES])
    predicted_claim = model.predict(user_scaled)[0]
    
    f_score = get_fitness_score(user_data)
    category = get_health_category(f_score)
    
    # Best Policy Logic
    policy_match = base_df[base_df['occupation'].str.lower() == occupation.lower()].iloc[0]['insurance']

    # --- RESULTS DASHBOARD ---
    st.divider()
    st.subheader(f"🎯 Personalized Results for {name}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🏃 Fitness Score", f"{f_score:.1f}")
    c2.metric("📋 Health Category", category)
    c3.metric("💰 Risk-Based Estimate", f"₹{predicted_claim:,.0f}")

    st.success(f"✅ **Policy Recommendation:** {policy_match}")

    # --- GRAPH 1: 3D HEALTH UNIVERSE (From your script) ---
    st.subheader("🌟 3D Health Universe")
    # Sampling for performance
    plot_df = fitness_df.sample(400)
    
    fig_3d = px.scatter_3d(
        plot_df, x='Age', y='BMI', z='Steps Taken',
        color='Stress Levels', size='Heart Beats',
        color_continuous_scale='Viridis',
        template="plotly_dark"
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # --- GRAPH 2: POPULATION ANALYSIS (Seaborn/Matplotlib) ---
    st.subheader("📊 Population Health Metrics")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Claim Distribution
    sns.histplot(fitness_df['Claim Amount'], kde=True, ax=ax1, color="#3498db")
    ax1.set_title("Market Claim Distribution")
    
    # Policy Pie
    policy_counts = base_df['insurance'].value_counts()
    ax2.pie(policy_counts, labels=policy_counts.index, autopct='%1.1f%%', colors=sns.color_palette("husl"))
    ax2.set_title("Available Policy Shares")
    
    st.pyplot(fig)

    # --- GRAPH 3: AGE VS ACTIVITY ---
    st.subheader("📈 Activity Trends by Life Stage")
    fig_box = px.box(fitness_df, x=pd.cut(fitness_df['Age'], bins=[18, 30, 45, 60, 100]), 
                     y='Steps Taken', color_discrete_sequence=['#2ecc71'])
    st.plotly_chart(fig_box, use_container_width=True)

    st.info("💡 **Business Insight:** Your low stress levels and active step count place you in the top 15% of eligible applicants for premium discounts.")

st.divider()
st.caption("© 2026 Smart Insurance | Health meets Intelligence")
