# =========================================
# SMART FITNESS INSURANCE – STREAMLIT APP
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Smart Fitness Insurance",
    page_icon="💙",
    layout="wide"
)

# -----------------------------------------
# CUSTOM THEME
# -----------------------------------------
st.markdown("""
<style>
.big-title {font-size:40px; font-weight:700; color:#0f4c75;}
.sub-title {font-size:18px; color:#3282b8;}
.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# LOAD & TRAIN MODEL (UNCHANGED IDEA)
# -----------------------------------------
@st.cache_data
def load_model():
    df = pd.read_csv("fitness_claim_dataset.csv").dropna()

    cat_cols = df.select_dtypes(include='object').columns.difference(['Name'])
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df["Fitness Score"] = (
        0.1*df['Blood Pressure (Systolic)'] +
        0.1*df['Blood Pressure (Diastolic)'] +
        0.15*df['Heart Beats'] +
        0.15*df['BMI'] +
        0.1*df['Cholesterol'] +
        0.2*df['Steps Taken'] +
        0.1*df['Active Minutes'] +
        0.1*df['Sleep Duration'] +
        0.05*df['Sleep Quality'] +
        0.15*df['VO2 Max'] +
        0.1*df['Calories Burned'] +
        0.15*df['SpO2 Levels'] -
        0.2*df['Stress Levels']
    )

    df["Fitness Score"] = (df["Fitness Score"] - df["Fitness Score"].min()) / \
                          (df["Fitness Score"].max() - df["Fitness Score"].min()) * 100

    X = df.drop(["Name", "Fitness Score"], axis=1)
    y = df["Fitness Score"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return df, model

df, model = load_model()

# -----------------------------------------
# HELPERS
# -----------------------------------------
def discount(score):
    if score >= 90: return 30
    elif score >= 80: return 25
    elif score >= 70: return 20
    elif score >= 60: return 15
    elif score >= 50: return 10
    elif score >= 40: return 5
    else: return 0

def category(score):
    if score >= 80: return "🏆 Elite"
    elif score >= 65: return "💪 Excellent"
    elif score >= 50: return "✅ Good"
    elif score >= 35: return "⚡ Fair"
    else: return "🔄 Needs Improvement"

def plan(d):
    if d >= 25: return "🌟 Premium Wellness Plan"
    elif d >= 15: return "💼 Standard Health Plan"
    elif d >= 5: return "🔰 Basic Fitness Cover"
    else: return "🛡️ Essential Protection"

# -----------------------------------------
# HEADER
# -----------------------------------------
st.markdown("<div class='big-title'>💙 Smart Fitness Insurance</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Personalized wellness, smarter insurance</div>", unsafe_allow_html=True)
st.divider()

# -----------------------------------------
# MEMBER QUESTION
# -----------------------------------------
member = st.radio("Are you an existing insurance member?", ["Yes", "No"], horizontal=True)

# =========================================
# MEMBER FLOW
# =========================================
if member == "Yes":
    st.subheader("👤 Member Details")

    name = st.selectbox("Select Name", df["Name"].unique())
    age = st.slider("Age", 18, 80, 30)

    user_row = df[(df["Name"] == name) & (df["Age"].round() == age)]

    if not user_row.empty:
        features = user_row.drop(["Name", "Fitness Score"], axis=1)
        score = model.predict(features)[0]
        disc = discount(score)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏃 Fitness Score", f"{score:.2f}")
        col2.metric("🎁 Discount", f"{disc}%")
        col3.metric("🏅 Category", category(score))
        col4.metric("📜 Plan", plan(disc))

        st.success(f"🎉 Congratulations {name}! You fall under **{category(score)}** category.")

        # Graphs
        st.subheader("📊 Your Fitness Analytics")
        fig1 = px.histogram(df, x="Fitness Score", nbins=30, title="Fitness Score Distribution")
        fig2 = px.scatter_3d(df, x="Age", y="BMI", z="Steps Taken",
                             color="Fitness Score", hover_name="Name")

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

        # Tips
        st.info("💡 **Personalized Tips:** Stay consistent with steps, improve sleep quality, manage stress.")

        # Certificate
        if st.button("📥 Download Fitness Certificate"):
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            c = canvas.Canvas(temp.name, pagesize=A4)
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(300, 800, "Fitness Insurance Certificate")
            c.setFont("Helvetica", 14)
            c.drawString(50, 700, f"Name: {name}")
            c.drawString(50, 670, f"Age: {age}")
            c.drawString(50, 640, f"Fitness Score: {score:.2f}")
            c.drawString(50, 610, f"Category: {category(score)}")
            c.drawString(50, 580, f"Insurance Plan: {plan(disc)}")
            c.save()

            with open(temp.name, "rb") as f:
                st.download_button("Download Certificate", f, file_name="fitness_certificate.pdf")

# =========================================
# NON-MEMBER FLOW
# =========================================
else:
    st.subheader("🧾 New Customer Details")

    name = st.text_input("Name")
    age = st.slider("Age", 18, 80, 30)
    income = st.selectbox("Income Range", ["<5 LPA", "5–10 LPA", "10–20 LPA", "20+ LPA"])
    habits = st.multiselect("Bad Habits (if any)", ["Smoking", "Alcohol", "Sedentary Lifestyle", "None"])

    st.subheader("💡 Recommended Plans For You")

    st.markdown("""
    ✔ **Basic Fitness Cover** – Affordable, essential protection  
    ✔ **Standard Health Plan** – Balanced coverage with wellness rewards  
    ✔ **Premium Wellness Plan** – Full health + fitness benefits  
    """)

    st.info("📞 **Get in touch with us**  
    ✉️ mail@insurance.gmail.com  
    📱 9812335644")

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.divider()
st.caption("© 2026 Smart Fitness Insurance | Health meets Intelligence")
