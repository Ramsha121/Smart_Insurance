# ===============================
# SMART FITNESS INSURANCE SYSTEM
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart Fitness Insurance",
    page_icon="💙",
    layout="wide"
)

# -------------------------------
# CUSTOM STYLING
# -------------------------------
st.markdown("""
<style>
body { background-color: #f4f9ff; }
.big-title { font-size: 40px; font-weight: 700; color: #0f4c75; }
.sub-title { font-size: 18px; color: #3282b8; }
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fitness_claim_dataset.csv")
    df = df.dropna()

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.difference(['Name'])
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df

df = load_data()

# -------------------------------
# FITNESS SCORE (UNCHANGED LOGIC)
# -------------------------------
df['Fitness Score'] = (
    -0.15 * (df['Blood Pressure (Systolic)'] - 120) / 20 +
    -0.15 * (df['Blood Pressure (Diastolic)'] - 80) / 10 +
    -0.10 * (df['Heart Beats'] - 70) / 30 +
    -0.15 * (df['BMI'] - 22) / 8 +
    -0.10 * (df['Cholesterol'] - 200) / 100 +
    0.20 * df['Steps Taken'] / 10000 +
    0.15 * df['Active Minutes'] / 60 +
    0.10 * (df['Sleep Duration'] - 7) / 2 +
    0.15 * df['Sleep Quality'] / 10 +
    0.20 * df['VO2 Max'] / 50 +
    0.10 * df['Calories Burned'] / 2000 +
    0.20 * df['SpO2 Levels'] / 100 +
    -0.25 * df['Stress Levels'] / 10
)

df['Fitness Score'] = (
    (df['Fitness Score'] - df['Fitness Score'].min()) /
    (df['Fitness Score'].max() - df['Fitness Score'].min())
) * 100

# -------------------------------
# FITNESS CATEGORY
# -------------------------------
def categorize_fitness(score):
    if score >= 80:
        return "🏆 Elite"
    elif score >= 65:
        return "💪 Excellent"
    elif score >= 50:
        return "✅ Good"
    elif score >= 35:
        return "⚡ Fair"
    else:
        return "🔄 Needs Improvement"

df["Fitness Category"] = df["Fitness Score"].apply(categorize_fitness)

# -------------------------------
# DISCOUNT & PLAN
# -------------------------------
def discount(score):
    if score >= 80:
        return 30
    elif score >= 65:
        return 20
    elif score >= 50:
        return 10
    else:
        return 0

def recommended_plan(d):
    if d >= 25:
        return "🌟 Premium Wellness Plan"
    elif d >= 15:
        return "💼 Standard Health Advantage Plan"
    elif d >= 5:
        return "🔰 Basic Fitness Cover"
    else:
        return "🛡️ Essential Protection Plan"

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='big-title'>💙 Smart Fitness-Based Insurance System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Personalized insurance decisions using fitness intelligence</div>", unsafe_allow_html=True)
st.divider()

# -------------------------------
# USER TYPE SELECTION
# -------------------------------
st.subheader("👋 Welcome! Let’s get started")
user_type = st.radio(
    "Are you already an insurance member?",
    ["Yes, I am a member", "No, I want to explore plans"]
)

# ==================================================
# 🟢 EXISTING MEMBER FLOW
# ==================================================
if user_type == "Yes, I am a member":

    st.sidebar.header("👤 Member Details")
    name = st.sidebar.selectbox("Select your name", df["Name"].unique())
    age = st.sidebar.slider("Age", 18, 80, 30)

    user_row = df[(df["Name"] == name) & (df["Age"] == age)]

    if not user_row.empty:
        score = user_row["Fitness Score"].values[0]
        category = categorize_fitness(score)
        disc = discount(score)
        plan = recommended_plan(disc)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🏃 Fitness Score", f"{score:.1f}/100")
        col2.metric("🏅 Category", category)
        col3.metric("🎁 Discount", f"{disc}%")
        col4.metric("📜 Your Plan", plan)

        if category == "🏆 Elite":
            st.success("🎉 Congratulations! You are ELITE. Your lifestyle is truly inspiring!")
        elif category == "💪 Excellent":
            st.info("👏 Excellent work! You’re very close to Elite status.")
        elif category == "⚡ Fair":
            st.warning("⚠️ Small improvements can unlock better benefits.")
        else:
            st.error("💙 Don’t worry — we’ll help you improve step by step.")

        # -------------------------------
        # MEMBER ANALYTICS
        # -------------------------------
        st.subheader("📊 Your Health Insights")

        fig1 = px.histogram(df, x="Fitness Score", nbins=25,
                            title="Fitness Score Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter_3d(
            df, x="Age", y="BMI", z="Steps Taken",
            color="Fitness Score",
            title="3D Health Landscape"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# 🔵 NON-MEMBER FLOW
# ==================================================
else:
    st.sidebar.header("🧾 Basic Details")
    name = st.sidebar.text_input("Your Name")
    age = st.sidebar.slider("Age", 18, 80, 25)
    income = st.sidebar.selectbox(
        "Income Range",
        ["Below ₹3 LPA", "₹3–6 LPA", "₹6–10 LPA", "Above ₹10 LPA"]
    )
    habits = st.sidebar.multiselect(
        "Do you have any habits?",
        ["Smoking", "Alcohol", "Sedentary Lifestyle", "Irregular Sleep"]
    )

    st.subheader(f"👤 Hello {name if name else 'there'}!")

    risk = "High" if len(habits) >= 2 else "Moderate" if habits else "Low"

    if risk == "Low":
        plan = "🌟 Premium Wellness Plan"
        st.success("✅ You are a low-risk individual. Premium plans suit you best!")
    elif risk == "Moderate":
        plan = "💼 Standard Health Advantage Plan"
        st.warning("⚠️ A balanced plan with wellness support is recommended.")
    else:
        plan = "🛡️ Essential Protection Plan"
        st.error("🚨 High-risk profile detected. Start with strong coverage.")

    st.markdown(f"""
    ### 📜 Recommended Plan
    **{plan}**

    ### 💡 Personalized Tips
    - Increase physical activity
    - Improve sleep routine
    - Reduce stress levels
    - Regular health check-ups
    """)

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.success("🎯 This system demonstrates real-world InsurTech + HealthTech intelligence")
