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
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Title
st.title("❤️ Heart Disease Prediction System")
st.markdown("---")

# Load data directly from GitHub (raw dataset)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/fedesoriano/heart-failure-prediction/master/heart.csv"
    df = pd.read_csv(url)
    
    # Encode categorical variables
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df

# Train models
@st.cache_resource
def train_models():
    df = load_data()
    
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
    
    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': trained_models['Random Forest'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return trained_models, scaler, pd.DataFrame(results), feature_importance, X_test, y_test

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose an option",
    ["🏠 Home", "📊 Model Performance", "🎯 Predict Heart Disease", "📈 Feature Analysis"]
)

# Load models
with st.spinner("Loading models..."):
    models, scaler, results_df, feature_importance, X_test, y_test = train_models()

if option == "🏠 Home":
    st.header("Welcome to Heart Disease Prediction System")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Dataset Size", "918 patients")
    with col2:
        st.metric("🔬 Features", "11 clinical parameters")
    with col3:
        st.metric("🎯 Models", "3 ML algorithms")
    
    st.markdown("---")
    st.subheader("📋 About")
    st.write("""
    This system uses Machine Learning to predict the likelihood of heart disease based on 
    clinical parameters. Enter patient information in the **Predict Heart Disease** section 
    to get instant risk assessment.
    """)
    
    # Show data preview
    df = load_data()
    st.subheader("📊 Sample Data")
    st.dataframe(df.head(10))

elif option == "📊 Model Performance":
    st.header("Model Performance Comparison")
    
    # Display metrics
    st.dataframe(results_df.style.highlight_max(axis=0))
    
    # Accuracy comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(results_df['Model'], results_df['Accuracy'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('Accuracy Score')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, results_df['Accuracy']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Best model confusion matrix
    st.subheader("Confusion Matrix - Best Model")
    best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    st.success(f"🏆 Best Model: **{best_model_name}** with {results_df['Accuracy'].max():.2%} accuracy")

elif option == "🎯 Predict Heart Disease":
    st.header("Patient Risk Assessment")
    st.write("Fill in the patient details below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", 
                                  ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    
    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    # Convert to numerical values
    sex_val = 1 if sex == "Male" else 0
    chest_pain_map = {"ATA": 1, "NAP": 2, "ASY": 0, "TA": 3}
    chest_pain_val = chest_pain_map[chest_pain]
    fasting_bs_val = 1 if fasting_bs == "Yes" else 0
    resting_ecg_map = {"Normal": 1, "ST": 2, "LVH": 0}
    resting_ecg_val = resting_ecg_map[resting_ecg]
    exercise_angina_val = 1 if exercise_angina == "Yes" else 0
    st_slope_map = {"Up": 2, "Flat": 1, "Down": 0}
    st_slope_val = st_slope_map[st_slope]
    
    # Create feature array
    input_features = np.array([[
        age, sex_val, chest_pain_val, resting_bp, cholesterol,
        fasting_bs_val, resting_ecg_val, max_hr, exercise_angina_val,
        oldpeak, st_slope_val
    ]])
    
    # Scale features
    input_scaled = scaler.transform(input_features)
    
    if st.button("🔍 Predict Risk", type="primary"):
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Get predictions from all models
        for model_name, model in models.items():
            prediction = model.predict(input_scaled)[0]
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{model_name}**")
            with col2:
                if prediction == 1:
                    st.write("⚠️ **High Risk**")
                else:
                    st.write("✅ **Low Risk**")
            
            # Get probability if available
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_scaled)[0][1]
                with col3:
                    st.write(f"Risk: {prob:.1%}")
        
        # Ensemble prediction (majority vote)
        predictions = [model.predict(input_scaled)[0] for model in models.values()]
        final_prediction = max(set(predictions), key=predictions.count)
        
        st.markdown("---")
        st.subheader("🎯 Final Recommendation")
        
        if final_prediction == 1:
            st.error("⚠️ **HIGH RISK OF HEART DISEASE**\n\nPlease consult a healthcare professional for further evaluation.")
        else:
            st.success("✅ **LOW RISK OF HEART DISEASE**\n\nMaintain a healthy lifestyle and regular check-ups.")

elif option == "📈 Feature Analysis":
    st.header("Feature Importance Analysis")
    
    # Display feature importance chart
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(10)
    ax.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 10 Most Important Features for Heart Disease Prediction')
    ax.invert_yaxis()
    st.pyplot(fig)
    
    # Display feature importance table
    st.subheader("Complete Feature Ranking")
    st.dataframe(feature_importance)
    
    # Feature descriptions
    with st.expander("📖 Click to view feature descriptions"):
        st.markdown("""
        - **Age**: Age of the patient
        - **Sex**: Gender (M: Male, F: Female)
        - **ChestPainType**: Type of chest pain (ATA, NAP, ASY, TA)
        - **RestingBP**: Resting blood pressure (mm Hg)
        - **Cholesterol**: Serum cholesterol (mg/dl)
        - **FastingBS**: Fasting blood sugar > 120 mg/dl (0: False, 1: True)
        - **RestingECG**: Resting electrocardiogram results (Normal, ST, LVH)
        - **MaxHR**: Maximum heart rate achieved
        - **ExerciseAngina**: Exercise-induced angina (Y: Yes, N: No)
        - **Oldpeak**: ST depression induced by exercise relative to rest
        - **ST_Slope**: Slope of the peak exercise ST segment (Up, Flat, Down)
        """)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Heart Disease Prediction System")
# Paste the entire app.py code here (the long code from above)
