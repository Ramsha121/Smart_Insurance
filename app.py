import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(page_title="Insurance Recommendation System", layout="wide", page_icon="🎯")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #2996c4; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .insight-box { background-color: #b8624d; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3; margin-bottom: 20px; }
    h1, h2, h3 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True) # ✅ Fixed!

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_preprocess_data():
    # Load dataset (adjust path as needed)
    try:
        df = pd.read_csv("base_plans.csv")
    except FileNotFoundError:
        # Fallback dummy data generation if file not found for demonstration
        st.error("Dataset 'base_plans.csv' not found. Please ensure it's in the directory.")
        return None

    # Preprocessing logic from your script
    df.dropna(inplace=True)
    cols_to_drop = ['perc_premium_paid_by_cash_credit','Count_3-6_months_late','Count_6-12_months_late',
                    'Count_more_than_12_months_late','application_underwriting_score','target',
                    'sourcing_channel','residence_area_type']
    df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)
    
    # Feature Engineering
    df['age_in_years'] = df['age_in_days'] // 365
    
    # Randomly assign occupations for analysis (as per original script)
    np.random.seed(42)
    occupations = ['Engineer', 'Doctor', 'Teacher', 'Clerk', 'Manager', 'Laborer', 'Mechanic', 'Driver']
    df['occupation'] = np.random.choice(occupations, size=len(df))
    df['job_type'] = df['occupation'].apply(lambda x: 'White-collar' if x in ['Engineer', 'Doctor', 'Teacher', 'Manager'] else 'Blue-collar')
    
    # Policy Allocation Logic
    df['policy'] = 'LIC'
    df.loc[(df['age_in_years'] >= 18) & (df['Income'] > 50000), 'policy'] = 'StarLite'
    df.loc[(df['age_in_years'] >= 18) & (df['Income'] <= 50000), 'policy'] = 'Maxbupa'
    
    return df

# --- MODEL TRAINING ---
@st.cache_resource
def train_model(df):
    le_occ = LabelEncoder()
    df_train = df.copy()
    df_train['occ_enc'] = le_occ.fit_transform(df_train['occupation'])
    
    X = df_train[['age_in_years', 'occ_enc', 'Income']]
    y = df_train['policy']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le_occ, acc

# Load Data
df = load_and_preprocess_data()

if df is not None:
    model, le_occ, accuracy = train_model(df)

    # --- SIDEBAR: USER INPUTS ---
    st.sidebar.header("🎯 Personal Recommendation")
    user_name = st.sidebar.text_input("Customer Name", "John Doe")
    user_age = st.sidebar.slider("Select Age", 1, 100, 30)
    user_income = st.sidebar.number_input("Annual Income (₹)", min_value=1000, value=75000, step=5000)
    user_occ = st.sidebar.selectbox("Occupation", le_occ.classes_)

    if st.sidebar.button("Predict Best Policy"):
        occ_enc = le_occ.transform([user_occ])[0]
        prediction = model.predict([[user_age, occ_enc, user_income]])[0]
        
        st.sidebar.success(f"Recommended: **{prediction}**")
        st.sidebar.info(f"Model Accuracy: {accuracy*100:.2f}%")

    # --- MAIN CONTENT ---
    st.title("🏢 Insurance Intelligence Dashboard")
    st.markdown("---")

    # KEY METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", f"{len(df):,}")
    m2.metric("Total Revenue", f"₹{df['premium'].sum():,.0f}")
    m3.metric("Avg Customer Value", f"₹{df['premium'].mean():,.0f}")
    m4.metric("Retention Proxy", f"{df['no_of_premiums_paid'].mean():.2f} pmts")

    # TABS FOR ANALYSIS
    tab1, tab2, tab3 = st.tabs(["📊 Visual Analytics", "💼 Business Insights", "🔍 Model Performance"])

    with tab1:
# --- VISUALIZATION 1 ---
        st.subheader("📊 VISUALIZATION 1: INCOME-AGE RELATIONSHIP WITH POLICY SEGMENTATION")
        fig1 = px.scatter(df, x="age_in_years", y="Income", color="policy", 
                         size="premium", hover_data=['occupation'],
                         title="Age vs Income by Policy Type",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 1. ADD THIS LINE (The calculation):
        income_age_corr = df['Income'].corr(df['age_in_years'])
        
        # 2. THEN CALL THE MARKDOWN:
        st.markdown(f"""
        <div class="insight-box">
        <b>🔍 INCOME-AGE RELATIONSHIP INSIGHTS:</b><br>
        • Correlation coefficient: {income_age_corr:.3f}<br>
        • Weak correlation - Age and income are largely independent<br>
        • <b>StarLite:</b> Avg Age {df[df['policy']=='StarLite']['age_in_years'].mean():.1f}y, Avg Income ₹{df[df['policy']=='StarLite']['Income'].mean():,.0f}<br>
        • <b>Maxbupa:</b> Avg Age {df[df['policy']=='Maxbupa']['age_in_years'].mean():.1f}y, Avg Income ₹{df[df['policy']=='Maxbupa']['Income'].mean():,.0f}
        </div>
        """, unsafe_allow_html=True)

        
    with tab2:
        st.header("📊 ADVANCED BUSINESS INSIGHTS")
        
        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            st.markdown("### 🔍 POLICY DISTRIBUTION BY JOB TYPE")
            ct = pd.crosstab(df['job_type'], df['policy'], margins=True)
            st.dataframe(ct)
            
            st.markdown("### 🔍 JOB TYPE POLICY PREFERENCES")
            for job in df['job_type'].unique():
                subset = df[df['job_type']==job]
                pref = subset['policy'].mode()[0]
                pct = (subset['policy']==pref).mean()*100
                st.write(f"• **{job} workers:** Preferred policy: {pref} ({pct:.1f}%)")

        with col_ins2:
            st.markdown("### 📊 OCCUPATION-WISE INCOME ANALYSIS")
            occ_income = df.groupby('occupation')['Income'].agg(['mean', 'median', 'count']).sort_values(by='mean', ascending=False)
            st.dataframe(occ_income.style.format({'mean': '₹{:,.0f}', 'median': '₹{:,.0f}'}))
            
            st.markdown("### 📈 REVENUE PER POLICY TYPE")
            rev = df.groupby('policy')['premium'].sum()
            st.bar_chart(rev)

    with tab3:
        st.header("🤖 Machine Learning Model Details")
        st.write(f"The system uses a **Random Forest Classifier** with **{accuracy*100:.2f}% accuracy** to predict policy eligibility based on customer demographics.")
        
        # Feature Importance
        importances = model.feature_importances_
        feat_importances = pd.Series(importances, index=['Age', 'Occupation', 'Income'])
        fig_imp = px.bar(feat_importances, title="Feature Importance in Policy Recommendation")
        st.plotly_chart(fig_imp)

    st.markdown("---")
    st.markdown("🎉 **COMPREHENSIVE INSURANCE ANALYSIS COMPLETED!**")
