# ======================================================
# 🤖 INTELLIGENT INSURANCE POLICY RECOMMENDATION SYSTEM
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Insurance Recommendation",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------------------------------
# THEME
# ------------------------------------------------------
st.markdown("""
<style>
.big-title {font-size:42px; font-weight:800; color:#0f4c75;}
.sub-title {font-size:18px; color:#3282b8;}
.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------
st.markdown("<div class='big-title'>🤖 Intelligent Insurance Policy Recommendation</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning powered insurance decision system</div>", unsafe_allow_html=True)
st.divider()

# ------------------------------------------------------
# FILE UPLOADER (CRITICAL FIX)
# ------------------------------------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload insurance_dataset.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("👈 Please upload the **insurance_dataset.csv** file to continue.")
    st.stop()

# ------------------------------------------------------
# LOAD & PREPARE DATA
# ------------------------------------------------------
@st.cache_data
def load_and_prepare_data(file):
    df = pd.read_csv(file)

    raw_count = df.shape[0]
    df = df.dropna()
    clean_count = df.shape[0]

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

    return df, X, y, encoders, raw_count, clean_count

df, X, y, encoders, raw_n, clean_n = load_and_prepare_data(uploaded_file)

# ------------------------------------------------------
# DATA SUMMARY
# ------------------------------------------------------
st.subheader("📊 Dataset Overview")
st.write(f"**Raw Records:** {raw_n}")
st.write(f"**Clean Records:** {clean_n}")
st.write(f"**Insurance Policies:** {df['insurance'].unique().tolist()}")

# ------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"🤖 Model trained successfully | Accuracy: **{acc:.2%}**")

# ------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------
st.subheader("🔍 Feature Importance")
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig_imp = px.bar(
    feat_imp.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Important Features"
)
st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------
# USER INPUT
# ------------------------------------------------------
st.divider()
st.subheader("💬 Interactive Insurance Recommendation")

occupation_list = encoders["occupation"].classes_.tolist()
occupation = st.selectbox("Select Occupation", occupation_list)

age = st.slider("Age (years)", 18, 100, 35)
income = st.number_input("Annual Income (₹)", min_value=20000, value=300000)

# ------------------------------------------------------
# PREDICTION
# ------------------------------------------------------
if st.button("🎯 Recommend Insurance Policy"):

    occ_enc = encoders["occupation"].transform([occupation])[0]

    sample = X.mean().to_frame().T
    sample["age_in_years"] = age
    sample["Income"] = income
    sample["occupation_encoded"] = occ_enc

    probs = model.predict_proba(sample)[0]
    policies = model.classes_

    result_df = pd.DataFrame({
        "Policy": policies,
        "Confidence (%)": np.round(probs * 100, 2)
    }).sort_values(by="Confidence (%)", ascending=False)

    best_policy = result_df.iloc[0]["Policy"]

    st.success(f"🏆 **Recommended Policy:** {best_policy}")

    # Confidence bar chart
    fig_prob = px.bar(
        result_df,
        x="Policy",
        y="Confidence (%)",
        title="Prediction Confidence for All Policies",
        color="Confidence (%)"
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # Alternatives
    st.subheader("📋 Alternative Options")
    st.dataframe(result_df)

# ------------------------------------------------------
# BUSINESS INSIGHTS
# ------------------------------------------------------
st.divider()
st.subheader("📈 Business Intelligence Dashboard")

policy_dist = df["insurance"].value_counts().reset_index()
policy_dist.columns = ["Policy", "Customers"]

fig_dist = px.pie(
    policy_dist,
    names="Policy",
    values="Customers",
    title="Insurance Policy Distribution"
)
st.plotly_chart(fig_dist, use_container_width=True)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.divider()
st.caption("© 2026 Intelligent Insurance Recommendation System | ML + Analytics")
