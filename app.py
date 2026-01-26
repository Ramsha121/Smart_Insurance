# ============================================================
# 🤖 INTELLIGENT INSURANCE POLICY RECOMMENDATION SYSTEM
# STREAMLIT VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Insurance Recommendation",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM STYLING
# ------------------------------------------------------------
st.markdown("""
<style>
.big-title {font-size:38px; font-weight:800; color:#0f4c75;}
.sub-title {font-size:18px; color:#3282b8;}
.card {
    background:#ffffff;
    padding:20px;
    border-radius:16px;
    box-shadow:0 6px 18px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("<div class='big-title'>🤖 Intelligent Insurance Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning powered policy selection & business insights</div>", unsafe_allow_html=True)
st.divider()

# ------------------------------------------------------------
# LOAD + PREPROCESS DATA
# ------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("insurance_dataset.csv")

    raw_count = df.shape[0]
    df = df.dropna()
    cleaned_count = df.shape[0]

    df["age_in_years"] = (df["age_in_days"] / 365).round(0)

    categorical_cols = [
        "occupation",
        "sourcing_channel",
        "residence_area_type",
        "Occupation_Type"
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        encoders[col] = le

    features = [
        "age_in_years",
        "Income",
        "premium",
        "no_of_premiums_paid",
        "perc_premium_paid_by_cash_credit",
        "application_underwriting_score",
        "occupation_encoded",
        "sourcing_channel_encoded",
        "residence_area_type_encoded",
        "Occupation_Type_encoded",
        "Count_3-6_months_late",
        "Count_6-12_months_late",
        "Count_more_than_12_months_late"
    ]

    X = df[features]
    y = df["insurance"]

    return df, X, y, encoders, raw_count, cleaned_count

df, X, y, encoders, raw_n, clean_n = load_and_prepare_data()

# ------------------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ------------------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------------------
with st.expander("📊 Dataset Overview", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", raw_n)
    col2.metric("After Cleaning", clean_n)
    col3.metric("Model Accuracy", f"{accuracy:.2%}")

    st.write("### Insurance Policy Distribution")
    policy_dist = df["insurance"].value_counts().reset_index()
    policy_dist.columns = ["Policy", "Customers"]
    st.plotly_chart(px.pie(
        policy_dist,
        names="Policy",
        values="Customers",
        hole=0.4
    ), use_container_width=True)

# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------
st.subheader("🔍 Feature Importance Analysis")
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.plotly_chart(
    px.bar(
        imp_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h"
    ),
    use_container_width=True
)

# ------------------------------------------------------------
# USER INTERACTION
# ------------------------------------------------------------
st.divider()
st.subheader("💬 Insurance Policy Recommendation")

member = st.radio("Are you an existing customer?", ["Yes", "No"], horizontal=True)

occupations = sorted(df["occupation"].unique())

if member == "Yes":
    name = st.selectbox("Select Customer Name", sorted(df["name"].unique()))
    user_row = df[df["name"] == name].iloc[0]

    st.info(f"""
    **Customer Profile**
    • Age: {int(user_row['age_in_years'])}  
    • Occupation: {user_row['occupation']}  
    • Income: ₹{int(user_row['Income']):,}
    """)

    user_features = X.loc[user_row.name:user_row.name]
    probs = model.predict_proba(user_features)[0]
    classes = model.classes_

else:
    age = st.slider("Age", 18, 80, 30)
    occupation = st.selectbox("Occupation", occupations)

    occ_income = df[df["occupation"] == occupation]["Income"]
    suggested_income = int(occ_income.mean())

    income = st.number_input(
        "Annual Income",
        min_value=10000,
        value=suggested_income,
        step=10000
    )

    premium = int(income * 0.05)

    user_input = {
        "age_in_years": age,
        "Income": income,
        "premium": premium,
        "no_of_premiums_paid": 5,
        "perc_premium_paid_by_cash_credit": 50,
        "application_underwriting_score": 70,
        "occupation_encoded": encoders["occupation"].transform([occupation])[0],
        "sourcing_channel_encoded": 0,
        "residence_area_type_encoded": 0,
        "Occupation_Type_encoded": 0,
        "Count_3-6_months_late": 0,
        "Count_6-12_months_late": 0,
        "Count_more_than_12_months_late": 0
    }

    user_df = pd.DataFrame([user_input])
    probs = model.predict_proba(user_df)[0]
    classes = model.classes_

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------
results = pd.DataFrame({
    "Policy": classes,
    "Confidence (%)": probs * 100
}).sort_values(by="Confidence (%)", ascending=False)

best_policy = results.iloc[0]

st.success(f"🏆 **Recommended Policy:** {best_policy['Policy']} ({best_policy['Confidence (%)']:.2f}%)")

st.write("### 🔎 Prediction Confidence")
st.plotly_chart(
    px.bar(
        results,
        x="Policy",
        y="Confidence (%)",
        color="Policy"
    ),
    use_container_width=True
)

# ------------------------------------------------------------
# POLICY INSIGHTS
# ------------------------------------------------------------
st.subheader(f"📋 Insights for {best_policy['Policy']}")

policy_df = df[df["insurance"] == best_policy["Policy"]]

col1, col2, col3 = st.columns(3)
col1.metric("Customers", policy_df.shape[0])
col2.metric("Avg Premium", f"₹{int(policy_df['premium'].mean()):,}")
col3.metric("Avg Income", f"₹{int(policy_df['Income'].mean()):,}")

st.write("### 👔 Top Occupations")
occ_dist = policy_df["occupation"].value_counts().head(5)
st.plotly_chart(
    px.bar(
        occ_dist,
        x=occ_dist.index,
        y=occ_dist.values
    ),
    use_container_width=True
)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()
st.caption("© 2026 Intelligent Insurance Recommendation System | ML + Business Intelligence")
