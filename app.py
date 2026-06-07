import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction System")
st.markdown("### Enter your medical information below")

# Load and train models
@st.cache_data
def load_and_train():
    url = "https://raw.githubusercontent.com/fedesoriano/heart-failure-prediction/master/heart.csv"
    df = pd.read_csv(url)
    
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models, scaler

with st.spinner("Loading AI models..."):
    models, scaler = load_and_train()

# Input Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Personal Information")
    age = st.slider("Age (years)", 20, 100, 50)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    
    st.subheader("❤️ Vitals")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)

with col2:
    st.subheader("🩺 Medical History")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Convert inputs
sex_val = 1 if sex == "M" else 0
chest_map = {"ATA": 1, "NAP": 2, "ASY": 0, "TA": 3}
chest_val = chest_map[chest_pain]
ecg_map = {"Normal": 1, "ST": 2, "LVH": 0}
ecg_val = ecg_map[resting_ecg]
angina_val = 1 if exercise_angina == "Y" else 0
slope_map = {"Up": 2, "Flat": 1, "Down": 0}
slope_val = slope_map[st_slope]

# Create input array
input_data = np.array([[
    age, sex_val, chest_val, resting_bp, cholesterol,
    fasting_bs, ecg_val, max_hr, angina_val, oldpeak, slope_val
]])
input_scaled = scaler.transform(input_data)

# PREDICT BUTTON
if st.button("🔍 PREDICT HEART DISEASE RISK", type="primary", use_container_width=True):
    st.markdown("---")
    st.subheader("📊 Results")
    
    # Get predictions
    lr_pred = models['Logistic Regression'].predict(input_scaled)[0]
    rf_pred = models['Random Forest'].predict(input_scaled)[0]
    svm_pred = models['SVM'].predict(input_scaled)[0]
    
    # Display results in a nice format
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Logistic Regression")
        if lr_pred == 1:
            st.error("⚠️ **Heart Disease Detected**")
        else:
            st.success("✅ **No Heart Disease**")
    
    with col2:
        st.markdown("### Random Forest")
        if rf_pred == 1:
            st.error("⚠️ **Heart Disease Detected**")
        else:
            st.success("✅ **No Heart Disease**")
    
    with col3:
        st.markdown("### SVM")
        if svm_pred == 1:
            st.error("⚠️ **Heart Disease Detected**")
        else:
            st.success("✅ **No Heart Disease**")
    
    # Final Recommendation
    st.markdown("---")
    st.subheader("🎯 Final Recommendation")
    
    # Majority vote
    votes = [lr_pred, rf_pred, svm_pred]
    final = 1 if sum(votes) >= 2 else 0
    
    if final == 1:
        st.error("""
        ### ⚠️ HIGH RISK OF HEART DISEASE
        
        **Recommendations:**
        - Consult a cardiologist immediately
        - Get a comprehensive cardiac evaluation
        - Monitor blood pressure and cholesterol
        - Adopt a heart-healthy lifestyle
        """)
    else:
        st.success("""
        ### ✅ LOW RISK OF HEART DISEASE
        
        **Recommendations:**
        - Maintain regular exercise
        - Eat a balanced diet
        - Get regular health check-ups
        - Keep monitoring your health
        """)
    
    # Show Random Forest probability
    rf_prob = models['Random Forest'].predict_proba(input_scaled)[0][1]
    st.progress(rf_prob)
    st.caption(f"Risk confidence score: {rf_prob:.1%} (based on Random Forest model)")
