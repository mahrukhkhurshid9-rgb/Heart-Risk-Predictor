# app.py - Main Streamlit application file
# Save this as 'app.py' in your project directory

import streamlit as st
import pandas as pd
import numpy as np
import pickle
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

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
    }
    .prediction-high-risk {
        background-color: #ff6b6b;
        color: white;
    }
    .prediction-low-risk {
        background-color: #51cf66;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Prediction System</h1>
        <p>Machine Learning Model for Heart Disease Risk Assessment</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔬 Model Training", "🎯 Make Prediction", "📈 Model Comparison", "ℹ️ About"])

# Load and cache data
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset"""
    # Using the same dataset from your notebook
    url = "https://raw.githubusercontent.com/your-repo/heart.csv"
    # Alternative: Use local file or direct download
    # For demo, I'll create sample data structure
    
    # Let's load from a reliable source
    df = pd.read_csv("https://raw.githubusercontent.com/kaggle/datasets/raw/master/fedesoriano/heart-failure-prediction/heart.csv")
    
    # Encode categorical variables
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

@st.cache_resource
def train_models():
    """Train and return all models"""
    df, _ = load_and_preprocess_data()
    
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
    
    # Get feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': trained_models['Random Forest'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return trained_models, scaler, results, feature_importance, X.columns, y_test, X_test

# Home Page
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Dataset Size", "918 patients", "12 features")
    
    with col2:
        st.metric("🎯 Target Classes", "2", "Heart Disease vs Normal")
    
    with col3:
        st.metric("🤖 Models", "3", "Logistic Regression, RF, SVM")
    
    st.markdown("---")
    
    st.subheader("📋 About the Dataset")
    st.write("""
    The dataset contains medical information from 918 patients with 11 clinical features used to predict 
    heart disease risk. The goal is to classify whether a patient has heart disease (1) or not (0).
    """)
    
    # Display feature descriptions
    st.subheader("🔍 Clinical Features")
    
    feature_descriptions = {
        "Age": "Age of the patient [years]",
        "Sex": "Sex of the patient [M: Male, F: Female]",
        "ChestPainType": "Chest pain type [ATA, NAP, ASY, TA]",
        "RestingBP": "Resting blood pressure [mm Hg]",
        "Cholesterol": "Serum cholesterol [mm/dl]",
        "FastingBS": "Fasting blood sugar > 120 mg/dl [0: False, 1: True]",
        "RestingECG": "Resting electrocardiogram results [Normal, ST, LVH]",
        "MaxHR": "Maximum heart rate achieved [Numeric value]",
        "ExerciseAngina": "Exercise-induced angina [Y: Yes, N: No]",
        "Oldpeak": "ST depression induced by exercise relative to rest",
        "ST_Slope": "Slope of the peak exercise ST segment [Up, Flat, Down]"
    }
    
    for feature, description in feature_descriptions.items():
        st.markdown(f"- **{feature}:** {description}")
    
    # Load and show data preview
    try:
        df, _ = load_and_preprocess_data()
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10))
    except:
        st.warning("⚠️ Unable to load dataset. Please ensure the data file is available.")

# Model Training Page
elif page == "🔬 Model Training":
    st.header("🔬 Model Training & Evaluation")
    
    with st.spinner("Training models... This may take a moment."):
        try:
            trained_models, scaler, results, feature_importance, features, y_test, X_test = train_models()
            
            # Display results
            st.subheader("📊 Model Performance Comparison")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']))
            
            # Performance metrics chart
            st.subheader("📈 Performance Metrics Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            x = np.arange(len(metrics))
            width = 0.25
            
            for i, model in enumerate(results_df['Model']):
                values = results_df[results_df['Model'] == model][metrics].values[0]
                ax.bar(x + i*width, values, width, label=model)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x + width)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            
            # Feature Importance
            st.subheader("🔑 Top 10 Most Important Features")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = feature_importance.head(10)
            ax.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance (Random Forest)')
            ax.invert_yaxis()
            st.pyplot(fig)
            
            # Confusion Matrix for Best Model
            st.subheader("🎯 Confusion Matrix - Best Model")
            best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
            best_model = trained_models[best_model_name]
            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix - {best_model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            st.info("Please make sure the dataset is available or adjust the data loading method.")

# Prediction Page
elif page == "🎯 Make Prediction":
    st.header("🎯 Heart Disease Risk Prediction")
    st.write("Enter patient information below to assess heart disease risk.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", 
                                   ["ATA (Asymptomatic)", "NAP (Non-Anginal Pain)", 
                                    "ASY (Atypical Angina)", "TA (Typical Angina)"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    
    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    # Convert categorical inputs to numerical
    sex_encoded = 1 if sex == "Male" else 0
    
    chest_pain_map = {"ATA (Asymptomatic)": 1, "NAP (Non-Anginal Pain)": 2, 
                      "ASY (Atypical Angina)": 0, "TA (Typical Angina)": 3}
    chest_pain_encoded = chest_pain_map[chest_pain]
    
    fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0
    
    resting_ecg_map = {"Normal": 1, "ST-T Abnormality": 2, "Left Ventricular Hypertrophy": 0}
    resting_ecg_encoded = resting_ecg_map[resting_ecg]
    
    exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
    
    st_slope_map = {"Up": 2, "Flat": 1, "Down": 0}
    st_slope_encoded = st_slope_map[st_slope]
    
    # Create feature array
    features = np.array([[
        age, sex_encoded, chest_pain_encoded, resting_bp, cholesterol, 
        fasting_bs_encoded, resting_ecg_encoded, max_hr, 
        exercise_angina_encoded, oldpeak, st_slope_encoded
    ]])
    
    # Train models and make prediction
    if st.button("🔍 Predict Heart Disease Risk", type="primary"):
        with st.spinner("Analyzing patient data..."):
            try:
                trained_models, scaler, results, _, _, _, _ = train_models()
                features_scaled = scaler.transform(features)
                
                st.subheader("📊 Prediction Results")
                
                # Make predictions with all models
                for model_name, model in trained_models.items():
                    prediction = model.predict(features_scaled)[0]
                    probability = None
                    
                    # Get probability if available
                    if hasattr(model, "predict_proba"):
                        probability = model.predict_proba(features_scaled)[0][1]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**{model_name}**")
                    
                    with col2:
                        if prediction == 1:
                            st.markdown("⚠️ **High Risk**")
                        else:
                            st.markdown("✅ **Low Risk**")
                    
                    with col3:
                        if probability:
                            st.markdown(f"Risk Score: {probability*100:.1f}%")
                    
                    # Add visual indicator
                    if prediction == 1:
                        st.progress(probability if probability else 0.7)
                    else:
                        st.progress(probability if probability else 0.3)
                
                # Ensemble prediction (majority vote)
                ensemble_predictions = [model.predict(features_scaled)[0] for model in trained_models.values()]
                ensemble_result = max(set(ensemble_predictions), key=ensemble_predictions.count)
                
                st.markdown("---")
                st.subheader("🎯 Final Recommendation")
                
                if ensemble_result == 1:
                    st.markdown("""
                        <div class="prediction-box prediction-high-risk">
                            <h2>⚠️ HIGH RISK OF HEART DISEASE</h2>
                            <p>Please consult a healthcare professional for further evaluation.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("📋 **Recommendations:**")
                    st.markdown("""
                        - Schedule a comprehensive cardiac evaluation
                        - Monitor blood pressure and cholesterol levels
                        - Consider lifestyle modifications (diet, exercise, stress management)
                        - Regular follow-up with your healthcare provider
                    """)
                else:
                    st.markdown("""
                        <div class="prediction-box prediction-low-risk">
                            <h2>✅ LOW RISK OF HEART DISEASE</h2>
                            <p>Maintain a healthy lifestyle and regular check-ups.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("📋 **Health Tips:**")
                    st.markdown("""
                        - Maintain regular physical activity
                        - Follow a balanced diet rich in fruits and vegetables
                        - Manage stress effectively
                        - Get regular health check-ups
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure models are trained properly.")

# Model Comparison Page
elif page == "📈 Model Comparison":
    st.header("📈 Model Comparison & Analysis")
    
    with st.spinner("Loading model results..."):
        try:
            _, _, results, feature_importance, _, _, _ = train_models()
            results_df = pd.DataFrame(results)
            
            # Detailed metrics table
            st.subheader("Detailed Performance Metrics")
            st.dataframe(results_df.set_index('Model'))
            
            # Radar chart for model comparison
            st.subheader("Model Performance Radar Chart")
            
            # Create radar chart
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='polar')
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            for model in results_df['Model']:
                values = results_df[results_df['Model'] == model][categories].values[0].tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title('Model Performance Comparison', size=20, y=1.1)
            st.pyplot(fig)
            
            # Feature importance table
            st.subheader("Feature Importance Analysis")
            st.dataframe(feature_importance)
            
            # Best model recommendation
            best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
            best_accuracy = results_df.loc[results_df['Accuracy'].idxmax(), 'Accuracy']
            
            st.success(f"""
            🏆 **Best Performing Model**: {best_model}
            - **Accuracy**: {best_accuracy*100:.2f}%
            - This model provides the most reliable predictions for heart disease risk assessment.
            """)
            
        except Exception as e:
            st.error(f"Error loading model comparison: {str(e)}")

# About Page
elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### Heart Disease Prediction System
    
    This application uses machine learning to assess the risk of heart disease based on clinical parameters.
    
    #### How It Works:
    1. **Model Training**: Three different ML models are trained on historical patient data
    2. **Feature Analysis**: Important medical indicators are identified
    3. **Risk Prediction**: New patient data is analyzed to predict disease risk
    4. **Ensemble Voting**: Multiple models provide a consensus prediction
    
    #### Models Used:
    - **Logistic Regression**: Linear model for binary classification
    - **Random Forest**: Ensemble of decision trees for robust predictions
    - **Support Vector Machine (SVM)**: Finds optimal hyperplane for classification
    
    #### Clinical Relevance:
    The model considers 11 clinical features including:
    - Demographics (Age, Sex)
    - Vital signs (Blood Pressure, Heart Rate)
    - Blood markers (Cholesterol, Blood Sugar)
    - ECG measurements
    - Exercise stress test results
    
    #### Disclaimer:
    ⚠️ This tool is for educational and research purposes only. It should not replace professional medical advice. Always consult with a qualified healthcare provider for medical decisions.
    
    #### Dataset Source:
    Heart Failure Prediction Dataset from Kaggle
    """)
    
    # Display technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        - **Framework**: Streamlit
        - **ML Libraries**: Scikit-learn, Pandas, NumPy
        - **Visualization**: Matplotlib, Seaborn
        - **Data Preprocessing**: StandardScaler, LabelEncoder
        - **Cross-Validation**: 5-fold CV for model validation
        - **Class Balancing**: class_weight='balanced' for imbalanced data
        """)
    
    with st.expander("📊 Model Performance Metrics"):
        st.markdown("""
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: Positive predictive value
        - **Recall**: Sensitivity or true positive rate
        - **F1-Score**: Harmonic mean of precision and recall
        - **Cross-Validation**: Validates model stability across different data splits
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        Made with ❤️ using Streamlit | Heart Disease Prediction System
    </div>
""", unsafe_allow_html=True)
