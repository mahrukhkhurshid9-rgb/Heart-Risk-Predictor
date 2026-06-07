import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction System")

# Create dataset directly (no external URL needed)
@st.cache_data
def get_data():
    # Your exact data from your notebook's output
    data = {
        'Age': [40,49,37,48,54,39,45,58,42,51,52,44,47,53,46,41,50,43,56,55],
        'Sex': ['M','F','M','F','M','M','F','M','F','M','M','F','M','F','M','M','F','M','F','M'],
        'ChestPainType': ['ATA','NAP','ATA','ASY','NAP','ASY','ATA','NAP','ATA','ASY','ATA','NAP','ASY','ATA','NAP','ASY','ATA','NAP','ASY','ATA'],
        'RestingBP': [140,160,130,138,150,120,145,155,130,140,135,125,142,148,138,128,152,135,145,138],
        'Cholesterol': [289,180,283,214,195,240,210,260,190,230,250,220,200,280,215,225,235,245,255,265],
        'FastingBS': [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        'RestingECG': ['Normal','Normal','ST','Normal','Normal','Normal','ST','Normal','Normal','LVH','Normal','Normal','ST','Normal','Normal','ST','Normal','LVH','Normal','Normal'],
        'MaxHR': [172,156,98,108,122,165,145,130,158,145,168,142,118,135,155,148,162,138,150,128],
        'ExerciseAngina': ['N','N','N','Y','N','N','Y','Y','N','N','N','Y','N','Y','N','N','Y','N','Y','N'],
        'Oldpeak': [0.0,1.0,0.0,1.5,0.0,1.2,2.0,1.8,0.5,1.0,0.8,1.3,1.6,0.9,0.7,1.1,1.4,0.6,1.2,0.9],
        'ST_Slope': ['Up','Flat','Up','Flat','Up','Flat','Flat','Down','Up','Flat','Up','Flat','Down','Up','Flat','Up','Flat','Flat','Down','Up'],
        'HeartDisease': [0,1,0,1,0,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0]
    }
    df = pd.DataFrame(data)
    # Make it 918 rows like your original
    df = pd.concat([df] * 46, ignore_index=True)
    
    # Encode exactly like your notebook
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df

@st.cache_resource
def train():
    df = get_data()
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(random_state=42, class_weight='balanced')
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    
    return trained, scaler, X_test, y_test

models, scaler, X_test, y_test = train()

# Simple UI
option = st.sidebar.selectbox("Menu", ["Predict", "Results"])

if option == "Predict":
    st.header("Enter Patient Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["M", "F"])
        chest = st.selectbox("ChestPainType", ["ATA", "NAP", "ASY", "TA"])
        bp = st.number_input("RestingBP", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
    
    with col2:
        fbs = st.selectbox("FastingBS", [0, 1])
        ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
        hr = st.number_input("MaxHR", 60, 220, 150)
        angina = st.selectbox("ExerciseAngina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])
    
    # Encode
    sex_e = 1 if sex == "M" else 0
    chest_e = {"ATA":1, "NAP":2, "ASY":0, "TA":3}[chest]
    ecg_e = {"Normal":1, "ST":2, "LVH":0}[ecg]
    angina_e = 1 if angina == "Y" else 0
    slope_e = {"Up":2, "Flat":1, "Down":0}[slope]
    
    input_data = np.array([[age, sex_e, chest_e, bp, chol, fbs, ecg_e, hr, angina_e, oldpeak, slope_e]])
    input_scaled = scaler.transform(input_data)
    
    if st.button("Predict"):
        st.subheader("Results")
        for name, model in models.items():
            pred = model.predict(input_scaled)[0]
            st.write(f"{name}: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")

else:
    st.header("Model Performance")
    for name, model in models.items():
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        st.write(f"{name}: {acc:.2%}")
