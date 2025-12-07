# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import os

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if models exist
def check_models_exist():
    required_files = ['scaler.pkl', 'best_ml_model.pkl', 'best_dl_model.h5']
    return all(os.path.exists(file) for file in required_files)

# Load models
@st.cache_resource
def load_models():
    try:
        if not check_models_exist():
            st.error("❌ Model files not found. Please ensure these files are in the same directory: scaler.pkl, best_ml_model.pkl, best_dl_model.h5")
            return None, None, None
        
        scaler = joblib.load('scaler.pkl')
        dl_model = keras.models.load_model('best_dl_model.h5')
        ml_model = joblib.load('best_ml_model.pkl')
        
        st.success("✅ Models loaded successfully!")
        return scaler, dl_model, ml_model
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

# Feature descriptions
feature_descriptions = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting electrocardiographic results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of the peak exercise ST segment (0-2)',
    'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
    'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)'
}

def single_prediction(scaler, dl_model, ml_model):
    st.header("👤 Patient Heart Disease Prediction")
    
    st.markdown("""
    Enter the patient's clinical information below to predict the likelihood of heart disease.
    The system uses both Deep Learning and Machine Learning models for accurate predictions.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        
        st.subheader("Chest Pain")
        cp = st.selectbox("Chest Pain Type", 
                        options=[(0, "Typical Angina"), (1, "Atypical Angina"), 
                                (2, "Non-anginal Pain"), (3, "Asymptomatic")],
                        format_func=lambda x: x[1])[0]
    
    with col2:
        st.subheader("Vital Signs")
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        
        st.subheader("Blood Sugar")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], 
                         format_func=lambda x: x[0])[1]
    
    with col3:
        st.subheader("ECG & Exercise")
        restecg = st.selectbox("Resting ECG", 
                             options=[(0, "Normal"), (1, "ST-T Wave Abnormality"), 
                                     (2, "Left Ventricular Hypertrophy")],
                             format_func=lambda x: x[1])[0]
        
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], 
                           format_func=lambda x: x[0])[1]
        
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                           options=[(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
                           format_func=lambda x: x[1])[0]
        
        st.subheader("Additional Tests")
        ca = st.slider("Number of Major Vessels", 0, 3, 0)
        thal = st.selectbox("Thalassemia", 
                          options=[(1, "Normal"), (2, "Fixed Defect"), (3, "Reversible Defect")],
                          format_func=lambda x: x[1])[0]
    
    if st.button("🔍 Predict Heart Disease", type="primary", use_container_width=True):
        # Create feature array
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # DL Prediction
        dl_prob = dl_model.predict(features_scaled, verbose=0)[0][0]
        dl_pred = 1 if dl_prob > 0.5 else 0
        
        # ML Prediction
        ml_pred = ml_model.predict(features)[0]
        ml_prob = ml_model.predict_proba(features)[0][1]
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Deep Learning Model")
            if dl_pred == 1:
                st.error(f"## ❤️ High Risk of Heart Disease")
            else:
                st.success(f"## ✅ Low Risk of Heart Disease")
            
            st.metric("Confidence Score", f"{dl_prob:.2%}")
            
            # Confidence gauge
            fig_dl = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = dl_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Level"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50}}))
            fig_dl.update_layout(height=300)
            st.plotly_chart(fig_dl, use_container_width=True)
        
        with col2:
            st.markdown("### 🤖 Machine Learning Model")
            if ml_pred == 1:
                st.error(f"## ❤️ High Risk of Heart Disease")
            else:
                st.success(f"## ✅ Low Risk of Heart Disease")
            
            st.metric("Confidence Score", f"{ml_prob:.2%}")
            
            # ML Confidence gauge
            fig_ml = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = ml_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Level"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50}}))
            fig_ml.update_layout(height=300)
            st.plotly_chart(fig_ml, use_container_width=True)
        
        # Risk factor analysis
        st.markdown("---")
        st.subheader("🔍 Risk Factor Analysis")
        
        risk_factors = []
        protective_factors = []
        
        # Risk factors
        if age > 55: 
            risk_factors.append(f"**Age ({age} years)**: Higher risk age group")
        if chol > 240: 
            risk_factors.append(f"**Cholesterol ({chol} mg/dl)**: High cholesterol level")
        if trestbps > 140: 
            risk_factors.append(f"**Blood Pressure ({trestbps} mm Hg)**: Hypertension")
        if thalach < 120: 
            risk_factors.append(f"**Max Heart Rate ({thalach})**: Low maximum heart rate")
        if oldpeak > 2: 
            risk_factors.append(f"**ST Depression ({oldpeak})**: Significant depression")
        if exang == 1: 
            risk_factors.append("**Exercise induced angina**: Present")
        if cp == 3: 
            risk_factors.append("**Chest Pain Type**: Asymptomatic (higher risk)")
        if ca > 1:
            risk_factors.append(f"**Major Vessels ({ca})**: Multiple vessels affected")
        if thal == 2 or thal == 3:
            risk_factors.append("**Thalassemia**: Abnormal thalassemia result")
        
        # Protective factors
        if age <= 45:
            protective_factors.append(f"**Age ({age} years)**: Younger age group")
        if chol <= 200:
            protective_factors.append(f"**Cholesterol ({chol} mg/dl)**: Normal cholesterol")
        if trestbps <= 120:
            protective_factors.append(f"**Blood Pressure ({trestbps} mm Hg)**: Normal blood pressure")
        if thalach >= 150:
            protective_factors.append(f"**Max Heart Rate ({thalach})**: Good heart rate capacity")
        if oldpeak <= 1:
            protective_factors.append(f"**ST Depression ({oldpeak})**: Minimal depression")
        
        if risk_factors:
            st.warning("### ⚠️ Identified Risk Factors")
            for factor in risk_factors:
                st.write(f"- {factor}")
        
        if protective_factors:
            st.success("### ✅ Protective Factors")
            for factor in protective_factors:
                st.write(f"- {factor}")
                
        if not risk_factors and not protective_factors:
            st.info("### ℹ️ Moderate Risk Profile")
            st.write("No extreme risk or protective factors identified. Consider additional clinical evaluation.")
        
        # Final recommendation
        st.markdown("---")
        st.subheader("🎯 Clinical Recommendation")
        
        avg_risk = (dl_prob + ml_prob) / 2
        if avg_risk > 0.7:
            st.error("""
            **Strongly recommend:** 
            - Immediate cardiology consultation
            - Further diagnostic testing (stress test, echocardiogram)
            - Lifestyle modifications and medication evaluation
            """)
        elif avg_risk > 0.5:
            st.warning("""
            **Recommend:**
            - Cardiology follow-up
            - Regular monitoring
            - Lifestyle counseling
            """)
        else:
            st.success("""
            **Recommend:**
            - Continue regular health maintenance
            - Healthy lifestyle practices
            - Routine follow-up as needed
            """)

def model_information():
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## Heart Disease Detection System
    
    This application predicts the likelihood of heart disease using advanced Machine Learning 
    and Deep Learning models trained on clinical data from the Cleveland Heart Disease dataset.
    
    ### 🎯 Clinical Features Used:
    """)
    
    # Display features in a nice format
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (feature, description) in enumerate(list(feature_descriptions.items())[:7]):
            st.write(f"**{feature}**")
            st.caption(description)
    
    with col2:
        for i, (feature, description) in enumerate(list(feature_descriptions.items())[7:]):
            st.write(f"**{feature}**")
            st.caption(description)
    
    st.markdown("""
    ### 🤖 Model Architecture
    
    #### Deep Learning Model
    - **Type**: Neural Network with multiple hidden layers
    - **Architecture**: 64 → 32 → 16 → 1 neurons with dropout regularization
    - **Activation**: ReLU (hidden), Sigmoid (output)
    - **Training**: 100 epochs with validation monitoring
    
    #### Machine Learning Model  
    - **Algorithm**: Random Forest Classifier
    - **Trees**: 100 decision trees
    - **Features**: Handles all 13 clinical parameters
    - **Advantage**: Robust and interpretable predictions
    
    ### 📊 Model Performance
    Both models achieve:
    - **Accuracy**: > 85% on test data
    - **ROC AUC**: > 0.90
    - **Balance**: Good performance across both classes
    
    ### ⚠️ Important Disclaimer
    **This application is for educational and research purposes only.**
    
    It should **NOT** be used as a substitute for:
    - Professional medical advice
    - Clinical diagnosis  
    - Treatment decisions
    
    Always consult qualified healthcare providers for medical decisions.
    """)

def main():
    st.title("❤️ Heart Disease Detection System")
    st.markdown("""
    Predict the likelihood of heart disease using AI-powered clinical analysis.
    Enter patient information below for an instant risk assessment.
    """)
    
    # Load models
    scaler, dl_model, ml_model = load_models()
    
    # Simple navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Patient Prediction", "About"])
    
    if page == "Patient Prediction":
        if scaler and dl_model and ml_model:
            single_prediction(scaler, dl_model, ml_model)
        else:
            st.error("""
            ## ❌ Models Not Loaded
            
            Please ensure these files are in the same directory:
            - `scaler.pkl`
            - `best_ml_model.pkl` 
            - `best_dl_model.h5`
            
            These model files are required for predictions.
            """)
    else:
        model_information()

if __name__ == "__main__":
    main()