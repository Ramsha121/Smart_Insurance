# ==========================================================
# SMART FITNESS–BASED INSURANCE RECOMMENDATION SYSTEM
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Smart Fitness Insurance",
    page_icon="💙",
    layout="wide"
)

# ----------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #0f4c75;
}
.sub-title {
    font-size: 18px;
    color: #3282b8;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fitness_claim_dataset.csv")
    df = df.dropna()

    cat_cols = df.select_dtypes(include=['object']).columns.difference(['Name'])
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Fitness Score (same philosophy)
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
        0.20 * df['SpO2 Levels'] / 100 -
        0.25 * df['Stress Levels'] / 10
    )

    df['Fitness Score'] = (
        (df['Fitness Score'] - df['Fitness Score'].min()) /
        (df['Fitness Score'].max() - df['Fitness Score'].min())
    ) * 100

    return df

df = load_data()

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
def predict_discount(score):
    if score >= 90: return 30
    elif score >= 80: return 25
    elif score >= 70: return 20
    elif score >= 60: return 15
    elif score >= 50: return 10
    elif score >= 40: return 5
    else: return 0

def categorize_fitness(score):
    if score >= 80: return "🏆 Elite"
    elif score >= 65: return "💪 Excellent"
    elif score >= 50: return "✅ Good"
    elif score >= 35: return "⚡ Fair"
    else: return "🔄 Needs Improvement"

def recommend_plan(discount):
    if discount >= 25:
        return "🌟 Premium Wellness Insurance"
    elif discount >= 15:
        return "💼 Standard Health Advantage Plan"
    elif discount >= 5:
        return "🔰 Basic Fitness Protection"
    else:
        return "🛡️ Essential Health Cover"

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown("<div class='big-title'>💙 Smart Fitness-Based Insurance System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Personalized Insurance Decisions Powered by Fitness Analytics</div>", unsafe_allow_html=True)
st.divider()

# ----------------------------------------------------------
# USER TYPE SELECTION
# ----------------------------------------------------------
st.subheader("👤 Are you an existing insurance member?")
user_type = st.radio("", ["Yes, I am a member", "No, I am not a member"])

# ==========================================================
# MEMBER FLOW
# ==========================================================
if user_type == "Yes, I am a member":

    st.subheader("🧾 Member Details")
    col1, col2 = st.columns(2)
    with col1:
        name = st.selectbox("Select your name", df["Name"].unique())
    with col2:
        age = st.slider("Age", 18, 80, 30)

    user_row = df[(df["Name"] == name) & (df["Age"] == age)]

    if not user_row.empty:
        score = user_row["Fitness Score"].values[0]
        discount = predict_discount(score)
        category = categorize_fitness(score)
        plan = recommend_plan(discount)

        st.divider()
        st.subheader("📊 Your Fitness & Insurance Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏃 Fitness Score", f"{score:.1f}/100")
        c2.metric("🎁 Discount", f"{discount}%")
        c3.metric("🏅 Category", category)
        c4.metric("📜 Plan", plan)

        if category == "🏆 Elite":
            st.success("🎉 Congratulations! You are an ELITE member. You enjoy the highest insurance benefits!")
        elif category == "💪 Excellent":
            st.info("👏 Excellent fitness! Maintain consistency to unlock elite rewards.")
        elif category == "✅ Good":
            st.warning("🙂 Good fitness level. Small improvements can increase your benefits.")
        else:
            st.error("💙 Your journey starts here. Follow healthy habits to improve coverage.")

        # Graphs
        st.divider()
        st.subheader("📈 Fitness Insights")

        fig1 = px.histogram(df, x="Fitness Score", nbins=25,
                            title="Fitness Score Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter_3d(
            df, x="Age", y="BMI", z="Steps Taken",
            color="Fitness Score",
            hover_name="Name",
            title="3D Health & Fitness Landscape"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# NON-MEMBER FLOW
# ==========================================================
else:
    st.subheader("📝 Basic Details (Non-Member)")
    name = st.text_input("Your Name")
    age = st.slider("Age", 18, 80, 30)
    income = st.selectbox("Income Range", ["< ₹3 LPA", "₹3–6 LPA", "₹6–10 LPA", "> ₹10 LPA"])
    habits = st.multiselect("Any bad habits?", ["Smoking", "Alcohol", "No regular exercise", "None"])

    st.divider()
    st.subheader("🧠 Insurance Recommendation")

    if income in ["> ₹10 LPA", "₹6–10 LPA"]:
        plan = "🌟 Premium Wellness Insurance"
    elif income == "₹3–6 LPA":
        plan = "💼 Standard Health Advantage Plan"
    else:
        plan = "🛡️ Essential Health Cover"

    st.success(f"✅ Recommended Plan for you: **{plan}**")

    if "Smoking" in habits or "Alcohol" in habits:
        st.warning("⚠️ Tip: Quitting bad habits can significantly reduce your future premium.")
    else:
        st.info("👏 Healthy lifestyle detected! You may qualify for discounts after enrollment.")

# ----------------------------------------------------------
# CONTACT SECTION
# ----------------------------------------------------------
st.divider()
st.subheader("📞 Get in Touch With Us")

st.markdown("""
📧 **Email:** mail@insurance.gmail.com  
📱 **Phone:** +91 98123 35644  

💙 *We’re here to help you build a healthier and safer future.*
""")

st.success("🎯 Smart Insurance Decisions Start With Smart Health Data")
