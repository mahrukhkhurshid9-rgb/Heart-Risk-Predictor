# ONLY using libraries from your Colab notebook
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

# Page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Prediction System")
st.markdown("---")

# EXACT same code as your Colab notebook for loading and training
@st.cache_data
def load_data():
    # Using the same dataset from your Colab
    url = "https://raw.githubusercontent.com/fedesoriano/heart-failure-prediction/master/heart.csv"
    df = pd.read_csv(url)
    
    # EXACT same encoding as your Colab notebook
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df

@st.cache_resource
def train_models():
    # EXACT same code as your Colab notebook
    df = load_data()
    
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # EXACT same models as your Colab notebook
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
    }
    
    trained_models = {}
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
    
    # Feature importance (EXACT same as your Colab)
    rf_model = trained_models['Random Forest']
    importances = rf_model.feature_importances_
    features = X.columns
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    return trained_models, scaler, pd.DataFrame(results), feat_df, X_test, y_test

# Train models (EXACT same as your Colab)
with st.spinner("Training models..."):
    models, scaler, results_df, feature_importance, X_test, y_test = train_models()

# Sidebar - EXACT same structure as your analysis
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose", ["🏠 Home", "📊 Model Performance", "🎯 Predict", "📈 Feature Importance"])

if option == "🏠 Home":
    st.header("Dataset Overview")
    df = load_data()
    st.write(f"**Dataset Shape:** {df.shape}")
    st.write("**First 5 rows:**")
    st.dataframe(df.head())
    st.write("**Target Distribution (0=No Disease, 1=Disease):**")
    st.bar_chart(df['HeartDisease'].value_counts())

elif option == "📊 Model Performance":
    st.header("Model Performance Comparison")
    
    # Display results table (EXACT same as your Colab output)
    st.dataframe(results_df)
    
    # Best model (EXACT same as your Colab)
    best = results_df.loc[results_df['Accuracy'].idxmax()]
    st.success(f"🏆 **Best Model:** {best['Model']} with {best['Accuracy']*100:.2f}% accuracy")
    
    # Confusion Matrix for Best Model (EXACT same as your Colab)
    best_model_name = best['Model']
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

elif option == "🎯 Predict":
    st.header("Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("ChestPainType", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("RestingBP", min_value=0, max_value=300, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
    
    with col2:
        fasting_bs = st.selectbox("FastingBS", [0, 1])
        resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("MaxHR", min_value=0, max_value=250, value=150)
        exercise_angina = st.selectbox("ExerciseAngina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
        st_slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])
    
    # Encode inputs EXACTLY like your Colab notebook
    sex_enc = 1 if sex == "M" else 0
    chest_map = {"ATA": 1, "NAP": 2, "ASY": 0, "TA": 3}
    chest_enc = chest_map[chest_pain]
    ecg_map = {"Normal": 1, "ST": 2, "LVH": 0}
    ecg_enc = ecg_map[resting_ecg]
    angina_enc = 1 if exercise_angina == "Y" else 0
    slope_map = {"Up": 2, "Flat": 1, "Down": 0}
    slope_enc = slope_map[st_slope]
    
    input_data = np.array([[
        age, sex_enc, chest_enc, resting_bp, cholesterol,
        fasting_bs, ecg_enc, max_hr, angina_enc, oldpeak, slope_enc
    ]])
    
    input_scaled = scaler.transform(input_data)
    
    if st.button("Predict"):
        st.subheader("Results")
        
        for name, model in models.items():
            pred = model.predict(input_scaled)[0]
            result = "⚠️ **Heart Disease**" if pred == 1 else "✅ **No Heart Disease**"
            st.write(f"**{name}:** {result}")

elif option == "📈 Feature Importance":
    st.header("Feature Importance Analysis")
    
    # EXACT same as your Colab notebook
    fig, ax = plt.subplots(figsize=(10, 6))
    top5 = feature_importance.head(5)
    ax.barh(top5['Feature'], top5['Importance'], color='steelblue')
    ax.set_title('Top 5 Important Features')
    ax.set_xlabel('Importance Score')
    ax.invert_yaxis()
    st.pyplot(fig)
    
    st.write("**Complete Feature Importance:**")
    st.dataframe(feature_importance)

st.markdown("---")
st.caption("Heart Disease Prediction System | Based on Random Forest (88.59% accuracy)")
