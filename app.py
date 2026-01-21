# ===============================
# SMART FITNESS INSURANCE DASHBOARD
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
    page_title="Smart Fitness Analytics",
    page_icon="💙",
    layout="wide"
)

# -------------------------------
# STYLING
# -------------------------------
sns.set_palette("husl")
plt.style.use("seaborn-v0_8")

st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: 700;
    color: #0f4c75;
}
.section {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='big-title'>💙 Comprehensive Fitness & Insurance Analytics</div>", unsafe_allow_html=True)
st.caption("End-to-end data exploration, visualization & wellness scoring")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fitness_claim_dataset.csv")
    return df.dropna()

df = load_data()

# -------------------------------
# DATA OVERVIEW
# -------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("📋 Dataset Overview")
st.write("**Shape:**", df.shape)
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# ENCODING
# -------------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns.difference(["Name"])
df_encoded = df.copy()
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le

numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns

# -------------------------------
# STATISTICS
# -------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("📈 Statistical Summary")
st.dataframe(df_encoded[numeric_cols].describe().round(2))
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# PAIRPLOT
# -------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("🔍 Fitness Relationships (Pairplot)")

selected_cols = ['Age', 'BMI', 'Steps Taken', 'Sleep Duration', 'Stress Levels']
fig = sns.pairplot(df_encoded[selected_cols], diag_kind="kde")
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# CORRELATION HEATMAP
# -------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("🔥 Correlation Heatmap")

plt.figure(figsize=(14, 10))
corr = df_encoded[numeric_cols].corr()
sns.heatmap(corr, cmap="RdYlBu_r", center=0, square=True)
st.pyplot(plt.gcf())
plt.clf()
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# BOXPLOTS
# -------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("💓 Health Metric Distributions")

metrics = [
    'Blood Pressure (Systolic)',
    'Blood Pressure (Diastolic)',
    'Heart Beats',
    'BMI'
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    sns.boxplot(y=df_encoded[metric], ax=ax)
    ax.set_title(metric)

st.pyplot(fig)
plt.clf()
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 3D PLOTLY
# -------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("🌐 3D Health Universe")

fig3d = px.scatter_3d(
    df_encoded,
    x="Age",
    y="BMI",
    z="Steps Taken",
    color="Stress Levels",
    size="Heart Beats",
    hover_name="Name"
)
st.plotly_chart(fig3d, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# FITNESS SCORE
# -------------------------------
df_encoded["Fitness Score"] = (
    -0.15 * (df_encoded['Blood Pressure (Systolic)'] - 120) / 20
    -0.15 * (df_encoded['Blood Pressure (Diastolic)'] - 80) / 10
    -0.10 * (df_encoded['Heart Beats'] - 70) / 30
    -0.15 * (df_encoded['BMI'] - 22) / 8
    -0.10 * (df_encoded['Cholesterol'] - 200) / 100
    +0.20 * df_encoded['Steps Taken'] / 10000
    +0.15 * df_encoded['Active Minutes'] / 60
    +0.10 * (df_encoded['Sleep Duration'] - 7) / 2
    +0.15 * df_encoded['Sleep Quality'] / 10
    +0.20 * df_encoded['VO2 Max'] / 50
    +0.10 * df_encoded['Calories Burned'] / 2000
    +0.20 * df_encoded['SpO2 Levels'] / 100
    -0.25 * df_encoded['Stress Levels'] / 10
)

df_encoded["Fitness Score"] = (
    (df_encoded["Fitness Score"] - df_encoded["Fitness Score"].min()) /
    (df_encoded["Fitness Score"].max() - df_encoded["Fitness Score"].min()) * 100
)

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

df_encoded["Fitness Category"] = df_encoded["Fitness Score"].apply(categorize_fitness)

# -------------------------------
# FITNESS DISTRIBUTION
# -------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("🏃 Fitness Score Distribution")

fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(df_encoded["Fitness Score"], kde=True, bins=25, ax=ax)
st.pyplot(fig)
plt.clf()

category_counts = df_encoded["Fitness Category"].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(category_counts, labels=category_counts.index, autopct="%1.1f%%")
st.pyplot(fig2)
plt.clf()
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# AGE GROUP ANALYSIS
# -------------------------------
df_encoded["Age Group"] = pd.cut(
    df_encoded["Age"],
    bins=[18, 30, 40, 50, 60, 100],
    labels=["18–29", "30–39", "40–49", "50–59", "60+"]
)

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.subheader("📊 Age-wise Fitness Trends")

fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(x="Age Group", y="Fitness Score", data=df_encoded, ax=ax)
st.pyplot(fig)
plt.clf()
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# SAVE DATA
# -------------------------------
df_encoded.to_csv("enhanced_fitness_analysis.csv", index=False)

st.success("✅ Analysis complete. Enhanced dataset saved.")
