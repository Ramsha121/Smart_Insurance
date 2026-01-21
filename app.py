# ===============================
# SMART FITNESS INSURANCE APP
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
# CUSTOM THEME
# -------------------------------
st.markdown("""
<style>
.big-title {font-size:40px; font-weight:700; color:#0f4c75;}
.sub-title {font-size:18px; color:#3282b8;}
.box {background:#fff; padding:20px; border-radius:15px;
      box-shadow:0px 4px 12px rgba(0,0,0,0.08);}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("fitness_claim_dataset.csv")
df = df.dropna()

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.difference(['Name'])
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# -------------------------------
# FITNESS SCORE (ORIGINAL LOGIC PRESERVED)
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
# CATEGORY, DISCOUNT, PLAN
# -------------------------------
def categorize_fitness(score):
    if score >= 80: return "🏆 Elite"
    elif score >= 65: return "💪 Excellent"
    elif score >= 50: return "✅ Good"
    elif score >= 35: return "⚡ Fair"
    else: return "🔄 Needs Improvement"

def predict_discount(score):
    if score >= 80: return 30
    elif score >= 65: return 20
    elif score >= 50: return 10
    else: return 0

def recommend_plan(discount):
    if discount >= 30:
        return "🌟 Premium Wellness Insurance"
    elif discount >= 20:
        return "💼 Standard Health Advantage Plan"
    elif discount >= 10:
        return "🔰 Basic Fitness Insurance"
    else:
        return "🛡️ Entry Protection Plan"

df['Fitness Category'] = df['Fitness Score'].apply(categorize_fitness)
df['Discount'] = df['Fitness Score'].apply(predict_discount)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='big-title'>💙 Smart Fitness Insurance System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-driven fitness scoring & insurance personalization</div>", unsafe_allow_html=True)
st.divider()

# -------------------------------
# MEMBER CHECK
# -------------------------------
member = st.radio("🛡️ Are you an existing insurance member?", ["Yes", "No"])

# ===============================
# IF MEMBER
# ===============================
if member == "Yes":
    st.subheader("👤 Member Details")

    name = st.text_input("Enter your name")
    age = st.slider("Enter your age", 18, 80, 30)

    if st.button("🔍 View My Insurance Dashboard"):
        user = df.iloc[(df['Age'] - age).abs().argsort()[:1]]
        score = user['Fitness Score'].values[0]
        category = categorize_fitness(score)
        discount = predict_discount(score)
        plan = recommend_plan(discount)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏃 Fitness Score", f"{score:.1f}")
        col2.metric("🏅 Category", category)
        col3.metric("🎁 Discount", f"{discount}%")
        col4.metric("📜 Plan", plan)

        if category == "🏆 Elite":
            st.success("🎉 Congratulations! You are an ELITE member. Keep inspiring others!")
        elif category == "💪 Excellent":
            st.info("👏 Excellent fitness! You’re very close to Elite.")
        elif category == "✅ Good":
            st.warning("🙂 Good fitness — small improvements can unlock higher rewards.")
        else:
            st.error("💙 Let’s improve together. We recommend guided wellness plans.")

        # Graphs
        st.subheader("📊 Your Fitness Insights")

        fig1 = px.histogram(df, x="Fitness Score", title="Fitness Score Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter(df, x="Fitness Score", y="Discount",
                          title="Fitness Score vs Discount",
                          hover_data=["Age"])
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.scatter_3d(df, x="Age", y="BMI", z="Steps Taken",
                             color="Fitness Score",
                             title="🌐 3D Health View")
        st.plotly_chart(fig3, use_container_width=True)

# ===============================
# IF NOT MEMBER
# ===============================
else:
    st.subheader("🆕 New User Details")

    name = st.text_input("Enter your name")
    age = st.slider("Enter your age", 18, 80, 30)
    income = st.selectbox("Income Range", ["Below 3 LPA", "3–6 LPA", "6–10 LPA", "10+ LPA"])
    habits = st.multiselect("Any bad habits?", ["Smoking", "Alcohol", "Sedentary Lifestyle", "None"])

    if st.button("📋 Get My Plan Recommendation"):
        st.info("🔍 Based on your inputs, here’s what we recommend:")

        if "Smoking" in habits or "Alcohol" in habits:
            plan = "🛡️ Preventive Care Insurance Plan"
            tip = "🚭 We recommend quitting harmful habits to unlock better premiums."
        elif income in ["6–10 LPA", "10+ LPA"]:
            plan = "🌟 Premium Wellness Insurance"
            tip = "🏃‍♂️ Ideal for long-term fitness & claim benefits."
        else:
            plan = "🔰 Basic Health Starter Plan"
            tip = "💙 Affordable coverage with upgrade options."

        st.success(f"📜 Recommended Plan: **{plan}**")
        st.write(f"💡 Tip: {tip}")

# -------------------------------
# CONTACT
# -------------------------------
st.divider()
st.subheader("📞 Get in Touch With Us")

st.markdown("""
📧 **Email:** mail@insurance.gmail.com  
📱 **Phone:** 9812335644  

We’re happy to guide you toward a healthier and safer future 💙
""")

st.success("✅ Smart Fitness Insurance App Ready for Demo, Viva & Deployment")
