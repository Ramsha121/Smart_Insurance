# ===============================
# SMART FITNESS INSURANCE APP
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart Fitness Insurance",
    page_icon="💙",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS (Theme)
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #f4f9ff;
}
.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #0f4c75;
}
.sub-title {
    font-size: 18px;
    color: #3282b8;
}
.metric-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA & MODEL (UNCHANGED LOGIC)
# -------------------------------
@st.cache_data
def load_data_and_model():
    df = pd.read_csv("fitness_claim_dataset.csv")
    df = df.dropna()

    # Encode insurance column
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    insurance_cols = [col for col in df.columns if "insur" in col.lower()]
    if insurance_cols:
        df[insurance_cols[0]] = le.fit_transform(df[insurance_cols[0]])
        scaler = StandardScaler()
        numerical_columns = df.select_dtypes(include=[np.number]).columns.difference(['Age'])
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # FITNESS SCORE (same logic)
    df['Fitness Score'] = (
        0.1 * df['Blood Pressure (Systolic)'] +
        0.1 * df['Blood Pressure (Diastolic)'] +
        0.15 * df['Heart Beats'] +
        0.15 * df['BMI'] +
        0.1 * df['Cholesterol'] +
        0.2 * df['Steps Taken'] +
        0.1 * df['Active Minutes'] +
        0.1 * df['Sleep Duration'] +
        0.05 * df['Sleep Quality'] +
        0.15 * df['VO2 Max'] +
        0.1 * df['Calories Burned'] +
        0.15 * df['SpO2 Levels'] -
        0.2 * df['Stress Levels']
    )

    df['Fitness Score'] = (
        (df['Fitness Score'] - df['Fitness Score'].min()) /
        (df['Fitness Score'].max() - df['Fitness Score'].min())
    ) * 100

    X = df.drop(['Name', 'Fitness Score'], axis=1)
    y = df['Fitness Score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return df, model

# -------------------------------
# DISCOUNT LOGIC (UNCHANGED)
# -------------------------------
def predict_discount(score):
    if score >= 90:
        return 30
    elif score >= 80:
        return 25
    elif score >= 70:
        return 20
    elif score >= 60:
        return 15
    elif score >= 50:
        return 10
    elif score >= 40:
        return 5
    else:
        return 0

# -------------------------------
# LOAD EVERYTHING
# -------------------------------
df, model = load_data_and_model()

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='big-title'>💙 Smart Fitness-Based Insurance System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered wellness scoring for personalized insurance benefits</div>", unsafe_allow_html=True)
st.divider()

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("👤 Customer Details")

selected_name = st.sidebar.selectbox("Select Name", df["Name"].unique())
selected_age = st.sidebar.slider("Age", 18, 80, 30)

# -------------------------------
# PREDICTION
# -------------------------------
row = df[(df["Name"] == selected_name) & (df["Age"] == selected_age)]

if not row.empty:
    features = row.drop(['Name', 'Fitness Score'], axis=1)
    score = model.predict(features)[0]
    discount = predict_discount(score)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("🏃 Fitness Score", f"{score:.2f}/100")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("🎁 Insurance Discount", f"{discount}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        risk = "Low Risk ✅" if score >= 70 else "Moderate ⚠️" if score >= 50 else "High Risk 🚨"
        st.metric("📊 Risk Category", risk)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.warning("No matching customer found.")

st.divider()

# -------------------------------
# INTERACTIVE VISUALS
# -------------------------------
st.subheader("📊 Interactive Health Analytics")

tab1, tab2, tab3 = st.tabs(["🏃 Fitness Distribution", "🎯 Score vs Discount", "🌐 3D Health View"])

with tab1:
    fig = px.histogram(
        df,
        x="Fitness Score",
        nbins=25,
        color_discrete_sequence=["#3282b8"],
        title="Fitness Score Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df["Predicted Discount"] = df["Fitness Score"].apply(predict_discount)
    fig = px.scatter(
        df,
        x="Fitness Score",
        y="Predicted Discount",
        color="Predicted Discount",
        size="Fitness Score",
        title="Fitness Score vs Insurance Discount",
        hover_data=["Name", "Age"]
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.scatter_3d(
        df,
        x="Age",
        y="BMI",
        z="Steps Taken",
        color="Fitness Score",
        size="Fitness Score",
        hover_name="Name",
        title="3D Health & Fitness Landscape"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# BUSINESS IMPACT
# -------------------------------
st.divider()
st.subheader("🏦 Why Insurance Companies Care")

st.markdown("""
✔ **Risk-based premium optimization**  
✔ **Rewards healthy customers with lower premiums**  
✔ **Encourages long-term wellness engagement**  
✔ **Reduces claim probability using predictive analytics**  
✔ **Data-driven underwriting decisions**
""")

st.success("✅ Project demonstrates real-world InsurTech + HealthTech integration")
