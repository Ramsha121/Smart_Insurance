# ================================
# SMART INSURANCE & FITNESS APP
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart Insurance Recommendation System",
    page_icon="📊",
    layout="wide"
)

st.title("🏥 Smart Insurance & Fitness Recommendation System")
st.markdown("AI-driven insurance insights with fitness-inspired analytics")

# -------------------------------
# LOAD DATA FROM GITHUB
# -------------------------------
DATA_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/base_plans.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# -------------------------------
# BASIC FEATURE ENGINEERING
# -------------------------------
df["age"] = df["age_in_days"] // 365
df["job_type"] = df["occupation"].apply(
    lambda x: "White-collar" if x in ["Doctor", "Engineer", "Manager", "Teacher"] else "Blue-collar"
)

# -------------------------------
# MODEL
# -------------------------------
features = ["age", "Income", "no_of_premiums_paid"]
X = df[features]
y = df["insurance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ======================================================
# VISUALIZATION 1: INCOME-AGE RELATIONSHIP (POLICY SEG)
# ======================================================
st.header("📊 Income-Age Relationship with Policy Segmentation")

corr = df["age"].corr(df["Income"])

fig1 = px.scatter(
    df,
    x="age",
    y="Income",
    color="insurance",
    size="premium",
    hover_data=["occupation"],
    title="Income vs Age by Insurance Policy",
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown(f"""
**INSIGHTS**
- Correlation coefficient: **{corr:.3f}**
- Weak correlation → Age and income are largely independent
""")

# ======================================================
# VISUALIZATION 2: COMPREHENSIVE FEATURE RELATIONSHIPS
# ======================================================
st.header("📊 Feature Relationship Analysis")

numeric_cols = ["age", "Income", "premium", "no_of_premiums_paid"]
fig2 = px.scatter_matrix(
    df,
    dimensions=numeric_cols,
    color="insurance",
    title="Feature Interaction Matrix"
)
st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# VISUALIZATION 3: INTERACTIVE OCCUPATION HIERARCHY
# ======================================================
st.header("📊 Occupation Hierarchy & Policy Preference")

occ_summary = df.groupby("occupation").agg(
    AvgIncome=("Income", "mean"),
    AvgPremium=("premium", "mean"),
    Customers=("insurance", "count")
).reset_index()

fig3 = px.bar(
    occ_summary,
    x="occupation",
    y="AvgIncome",
    color="AvgPremium",
    text="Customers",
    title="Occupation-wise Income & Premium Analysis"
)
st.plotly_chart(fig3, use_container_width=True)

# ======================================================
# VISUALIZATION 4: ADVANCED CORRELATION MATRIX
# ======================================================
st.header("📊 Advanced Correlation Matrix")

corr_matrix = df[numeric_cols].corr()

fig4 = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu",
    title="Correlation Matrix"
)
st.plotly_chart(fig4, use_container_width=True)

# ======================================================
# VISUALIZATION 5: COMPREHENSIVE PREMIUM ANALYSIS
# ======================================================
st.header("📊 Comprehensive Premium Analysis")

premium_stats = df.groupby("insurance").agg(
    AvgPremium=("premium", "mean"),
    TotalRevenue=("premium", "sum"),
    AvgIncome=("Income", "mean"),
    CompletionRate=("no_of_premiums_paid", "mean")
).reset_index()

fig5 = px.bar(
    premium_stats,
    x="insurance",
    y="TotalRevenue",
    color="AvgPremium",
    title="Policy-wise Revenue & Premium Performance",
    text="CompletionRate"
)
st.plotly_chart(fig5, use_container_width=True)

# ======================================================
# BUSINESS INTELLIGENCE DASHBOARD
# ======================================================
st.header("📊 Business Intelligence Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", len(df))
c2.metric("Total Revenue", f"₹{df['premium'].sum():,.0f}")
c3.metric("Avg Premium", f"₹{df['premium'].mean():,.0f}")
c4.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# ======================================================
# POLICY DISTRIBUTION BY JOB TYPE
# ======================================================
st.header("📊 Policy Distribution by Job Type")

job_policy = pd.crosstab(df["job_type"], df["insurance"])

fig6 = px.bar(
    job_policy,
    barmode="group",
    title="Policy Preference by Job Type"
)
st.plotly_chart(fig6, use_container_width=True)

# ======================================================
# INTERACTIVE RECOMMENDATION ENGINE
# ======================================================
st.header("🎯 Get Your Personalized Insurance Recommendation")

age = st.slider("Age", 18, 100, 30)
income = st.number_input("Annual Income", value=1000000)
occupation = st.selectbox("Occupation", df["occupation"].unique())
premiums_paid = st.slider("Premiums Paid", 1, 20, 10)

if st.button("Predict Insurance Policy"):
    input_df = pd.DataFrame(
        [[age, income, premiums_paid]],
        columns=features
    )

    probs = model.predict_proba(input_df)[0]
    labels = model.classes_

    result = pd.DataFrame({
        "Policy": labels,
        "Confidence (%)": probs * 100
    }).sort_values("Confidence (%)", ascending=False)

    st.success(f"Recommended Policy: {result.iloc[0]['Policy']}")
    st.dataframe(result, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown(
    "Smart Insurance Recommendation System | Data-Driven | Fitness-Inspired"
)
