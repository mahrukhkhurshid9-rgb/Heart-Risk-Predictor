import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction System")

# Load your saved models from Colab
@st.cache_resource
def load_my_models():
    # Load the models you saved from Colab
    with open('logistic_model.pkl', 'rb') as f:
        logistic_model = pickle.load(f)
    
    with open('randomforest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    models = {
        'Logistic Regression': logistic_model,
        'Random Forest': rf_model,
        'SVM': svm_model
    }
    
    return models, scaler

# Load models
try:
    models, scaler = load_my_models()
    st.success("✅ Models loaded successfully from your Colab training!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.info("Please make sure you have uploaded logistic_model.pkl, randomforest_model.pkl, svm_model.pkl, and scaler.pkl to your GitHub repository.")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Prediction", "Model Performance"])

if option == "Prediction":
    st.header("Enter Patient Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting BP", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    
    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    # Convert to numbers (SAME encoding as your Colab)
    sex_val = 1 if sex == "Male" else 0
    chest_map = {"ATA": 1, "NAP": 2, "ASY": 0, "TA": 3}
    chest_val = chest_map[chest_pain]
    fasting_val = 1 if fasting_bs == "Yes" else 0
    ecg_map = {"Normal": 1, "ST": 2, "LVH": 0}
    ecg_val = ecg_map[resting_ecg]
    angina_val = 1 if exercise_angina == "Yes" else 0
    slope_map = {"Up": 2, "Flat": 1, "Down": 0}
    slope_val = slope_map[st_slope]
    
    # Create input array
    input_data = np.array([[age, sex_val, chest_val, resting_bp, cholesterol, 
                            fasting_val, ecg_val, max_hr, angina_val, oldpeak, slope_val]])
    
    # Scale using YOUR scaler from Colab
    input_scaled = scaler.transform(input_data)
    
    if st.button("Predict Heart Disease Risk", type="primary"):
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Get predictions from YOUR trained models
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred = models['Logistic Regression'].predict(input_scaled)[0]
            st.metric("Logistic Regression", "⚠️ HIGH RISK" if pred == 1 else "✅ LOW RISK")
        
        with col2:
            pred = models['Random Forest'].predict(input_scaled)[0]
            st.metric("Random Forest", "⚠️ HIGH RISK" if pred == 1 else "✅ LOW RISK")
        
        with col3:
            pred = models['SVM'].predict(input_scaled)[0]
            st.metric("SVM", "⚠️ HIGH RISK" if pred == 1 else "✅ LOW RISK")
        
        # Get Random Forest probability (since it's your best model)
        rf_prob = models['Random Forest'].predict_proba(input_scaled)[0][1]
        st.progress(rf_prob)
        st.caption(f"Risk Probability: {rf_prob:.1%}")
        
        st.markdown("---")
        st.info("💡 This prediction is based on models trained in Google Colab with 88.59% accuracy")

else:
    st.header("Model Performance (from your Colab training)")
    
    st.write("""
    ### Your Model Results from Colab:
    
    | Model | Accuracy | Precision | Recall | F1-Score |
    |-------|----------|-----------|--------|----------|
    | Logistic Regression | 85.33% | 85.71% | 88.24% | 86.96% |
    | Random Forest | **88.59%** | **89.32%** | 90.20% | **89.76%** |
    | SVM | 87.50% | 86.92% | **91.18%** | 89.00% |
    
    ### Best Model: Random Forest with 88.59% accuracy
    
    ### Top 5 Important Features:
    1. ST_Slope (24.9%)
    2. MaxHR (11.5%)
    3. Cholesterol (11.5%)
    4. Oldpeak (11.1%)
    5. ChestPainType (10.7%)
    """)

st.markdown("---")
st.caption("Powered by models trained in Google Colab")
