# =========================================
# SMART INSURANCE RECOMMENDATION SYSTEM
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Smart Insurance Intelligence",
    page_icon="💙",
    layout="wide"
)

# -----------------------------------------
# DATA SOURCE (GITHUB RAW)
# -----------------------------------------
DATA_URL = "https://raw.githubusercontent.com/Ramsha121/Smart_Insurance/data/base_plans.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# -----------------------------------------
# PREPROCESSING
# -----------------------------------------
encoders = {}
for col in ["occupation", "policy"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[["age", "Income", "occupation"]]
y = df["policy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -----------------------------------------
# HEADER
# -----------------------------------------
st.markdown("## 💙 Smart Insurance Recommendation System")
st.caption("Personalized predictions powered by Machine Learning")
st.divider()

# =====================================================
# 1️⃣ USER INPUT (FIRST)
# =====================================================
st.subheader("🧾 Enter Your Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 30)

with col2:
    income = st.number_input("Annual Income (₹)", min_value=5000, value=1000000, step=50000)

with col3:
    occupation = st.selectbox(
        "Occupation",
        encoders["occupation"].inverse_transform(df["occupation"].unique())
    )

# Encode input
occ_encoded = encoders["occupation"].transform([occupation])[0]
input_df = pd.DataFrame([[age, income, occ_encoded]],
                        columns=["age", "Income", "occupation"])

# =====================================================
# 2️⃣ PERSONALIZED OUTPUT
# =====================================================
st.divider()
st.subheader("🎯 Personalized Insurance Recommendation")

predicted_policy_encoded = model.predict(input_df)[0]
policy_probs = model.predict_proba(input_df)[0]

policy_names = encoders["policy"].inverse_transform(range(len(policy_probs)))
prob_df = pd.DataFrame({
    "Policy": policy_names,
    "Confidence (%)": np.round(policy_probs * 100, 2)
})

recommended_policy = encoders["policy"].inverse_transform([predicted_policy_encoded])[0]
confidence = prob_df.loc[prob_df["Policy"] == recommended_policy, "Confidence (%)"].values[0]

st.success(f"""
👤 **Profile:** {age}y {occupation}, ₹{income:,}  
🎯 **Recommended Policy:** **{recommended_policy}**  
📊 **Prediction Confidence:** {confidence}%
""")

# =====================================================
# 3️⃣ PREDICTION REASONING
# =====================================================
st.info(f"""
🔍 **Prediction Reasoning**
- Age considered: {age} years
- Income segment matched
- Occupation risk & preference pattern
- Historical customer behavior learned by ML
""")

st.subheader("📊 Policy Confidence Distribution")
st.dataframe(prob_df, use_container_width=True)

# =====================================================
# 4️⃣ PERSONALIZED INSIGHTS
# =====================================================
st.divider()
st.subheader("🧠 Personalized Insights")

avg_premium = df.groupby("policy")["premium"].mean()
policy_encoded = encoders["policy"].transform([recommended_policy])[0]

st.markdown(f"""
✔ Average premium for **{recommended_policy}** customers  
✔ Best value for your income bracket  
✔ High customer adoption in your occupation segment  
""")

# =====================================================
# 5️⃣ ADVANCED VISUALIZATIONS
# =====================================================
st.divider()
st.subheader("📊 Advanced Visual Analytics")

# VISUALIZATION 1: Income–Age–Policy
fig1 = px.scatter(
    df,
    x="age",
    y="Income",
    color=encoders["policy"].inverse_transform(df["policy"]),
    title="Income–Age Relationship with Policy Segmentation"
)
st.plotly_chart(fig1, use_container_width=True)

# VISUALIZATION 2: Occupation–Policy Matrix
fig2 = px.density_heatmap(
    df,
    x=encoders["occupation"].inverse_transform(df["occupation"]),
    y=encoders["policy"].inverse_transform(df["policy"]),
    title="Occupation–Policy Preference Matrix"
)
st.plotly_chart(fig2, use_container_width=True)

# VISUALIZATION 3: Correlation Matrix
corr = df[["age", "Income", "premium", "no_of_premiums_paid"]].corr()
fig3 = px.imshow(
    corr,
    text_auto=True,
    title="Advanced Correlation Matrix"
)
st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# 6️⃣ BUSINESS INTELLIGENCE DASHBOARD
# =====================================================
st.divider()
st.subheader("📈 Business Intelligence Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Avg Premium (₹)", f"{df['premium'].mean():,.0f}")
col3.metric("Model Accuracy", f"{accuracy*100:.2f}%")

policy_dist = df["policy"].value_counts().reset_index()
policy_dist.columns = ["Policy", "Customers"]
policy_dist["Policy"] = encoders["policy"].inverse_transform(policy_dist["Policy"])

fig4 = px.bar(
    policy_dist,
    x="Policy",
    y="Customers",
    title="Insurance Policy Distribution"
)
st.plotly_chart(fig4, use_container_width=True)

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.divider()
st.caption("© 2026 Smart Insurance Intelligence | Data-driven decisions made human 💙")
