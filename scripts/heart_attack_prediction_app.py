import streamlit as st
import pandas as pd
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤",
    layout="wide"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HeartAttackPredictor:
    """A class for predicting heart attack risk using ensemble models."""
    
    def __init__(self, model_path='models/'):
        """Initialize predictor with path to models."""
        self.model_path = Path(model_path)
        self._setup_logging()
        self._load_components()
        
        # Define required features for prediction
        self.features = ['HadCOPD', 'AlcoholDrinkers', 'SmokerStatus', 'GeneralHealth', 
                        'DifficultyWalking', 'HadDiabetes', 'HighBloodPressure', 
                        'HadStroke', 'BlindOrVisionDifficulty', 'AgeCategory', 
                        'HighCholesterol', 'HadKidneyDisease', 'Sex', 'HadAngina']

    def _setup_logging(self):
        """Configure logging."""
        self.logger = logger

    def _load_components(self):
        """Load models and mappings."""
        try:
            # Load models
            self.models = {
                'xgb': self._load_model('best_xgb.pkl'),
                'rf': self._load_model('best_rf.pkl'),
                'lr': self._load_model('best_lr.pkl')
            }
            
            # Define mappings
            self._define_mappings()
            
            self.logger.info("Successfully loaded all components")
        except Exception as e:
            self.logger.error(f"Error loading components: {str(e)}")
            st.error(f"Error loading prediction models: {str(e)}")
            raise

    def _load_model(self, filename):
        """Load a single model from file."""
        try:
            with open(self.model_path / filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {str(e)}")
            raise

    def _define_mappings(self):
        """Define encoding mappings."""
        self.ordinal_mapping = {
            'GeneralHealth': {
                "Excellent": 5, "Very good": 4, "Good": 3, 
                "Fair": 2, "Poor": 1
            },
            'SmokerStatus': {
                "Current smoker - now smokes every day": 4,
                "Current smoker - now smokes some days": 3,
                "Former smoker": 2, 
                "Never smoked": 1
            },
            'AgeCategory': {
                "Age 18 to 24": 1, "Age 25 to 29": 2,
                "Age 30 to 34": 3, "Age 35 to 39": 4,
                "Age 40 to 44": 5, "Age 45 to 49": 6,
                "Age 50 to 54": 7, "Age 55 to 59": 8,
                "Age 60 to 64": 9, "Age 65 to 69": 10,
                "Age 70 to 74": 11, "Age 75 to 79": 12,
                "Age 80 or older": 13
            },
            'HadDiabetes': {
                'No': 1, 
                'No, pre-diabetes or borderline diabetes': 2,
                'Yes, but only during pregnancy (female)': 3, 
                'Yes': 4
            },
            'HighBloodPressure': {
                'No': 1,
                'No, pre-hypertension or borderline high blood pressure': 2,
                'Yes, but female told only during pregnancy': 3,
                'Yes': 4
            }
        }
        self.binary_mapping = {'No': 0, 'Yes': 1}

    def _validate_input(self, df):
        """Validate input data format and required columns."""
        missing_cols = set(self.features) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _preprocess_input(self, data):
        """Preprocess input data for prediction."""
        df = data.copy()
        
        # Apply binary encoding
        binary_cols = ['HadAngina', 'HadStroke', 'HighCholesterol',
                    'HadCOPD', 'DifficultyWalking', 'HadKidneyDisease', 
                    'BlindOrVisionDifficulty', 'AlcoholDrinkers']
        for col in binary_cols:
            df[col] = df[col].map(self.binary_mapping)
        
        # Apply ordinal encoding
        for col, mapping in self.ordinal_mapping.items():
            df[col] = df[col].map(mapping)
        
        # Encode Sex
        df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
        
        # Ensure correct column order to match the model's expected features
        df = df[self.features]
        
        return df

    def predict(self, data, threshold=0.5):
        """Make ensemble prediction."""
        try:
            self._validate_input(data)
            processed_data = self._preprocess_input(data)
            
            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                pred_proba = model.predict_proba(processed_data)[:, 1]
                predictions[name] = pred_proba[0]
                self.logger.info(f"{name} prediction: {pred_proba[0]:.3f}")

            # Calculate ensemble prediction
            final_proba = np.mean(list(predictions.values()))
            final_pred = int(final_proba > threshold)
            
            # Get risk analysis
            risk_factors = self._identify_risk_factors(data)
            risk_level = self._get_risk_level(final_proba)
            
            return {
                'prediction': bool(final_pred),
                'probability': float(final_proba),
                'risk_factors': risk_factors,
                'risk_level': risk_level,
                'model_predictions': predictions
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

    def _identify_risk_factors(self, data):
        """Identify main risk factors from input data."""
        risk_factors = []
        
        # Check age risk
        if data['AgeCategory'].iloc[0] in ['Age 65 to 69', 'Age 70 to 74', 
                                           'Age 75 to 79', 'Age 80 or older']:
            risk_factors.append('Advanced Age')
        
        # Check health conditions
        conditions = {
            'HighBloodPressure': 'High Blood Pressure',
            'HighCholesterol': 'High Cholesterol',
            'HadDiabetes': 'Diabetes',
            'HadAngina': 'Angina',
            'HadStroke': 'Previous Stroke',
            'HadCOPD': 'COPD',
            'HadKidneyDisease': 'Kidney Disease'
        }
        
        for col, condition in conditions.items():
            if data[col].iloc[0] == 'Yes':
                risk_factors.append(condition)
        
        # Check other risk factors
        if data['DifficultyWalking'].iloc[0] == 'Yes':
            risk_factors.append('Difficulty Walking')
            
        if data['GeneralHealth'].iloc[0] in ['Poor', 'Fair']:
            risk_factors.append('Poor General Health')
            
        if data['SmokerStatus'].iloc[0].startswith('Current smoker'):
            risk_factors.append('Current Smoker')
            
        return risk_factors

    def _get_risk_level(self, probability):
        """Convert probability to risk level."""
        if probability < 0.2:
            return 'Low Risk'
        elif probability < 0.4:
            return 'Moderate Risk'
        elif probability < 0.6:
            return 'High Risk'
        else:
            return 'Very High Risk'

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
    st.warning("Model directory created. Please place your trained models in the 'models' folder.")

# Main application code
def main():
    # Title section
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    with col2:
        st.title("Heart Attack Risk Predictor")
        st.write("A machine learning tool to assess your risk of heart attack")

    # Create sidebar for inputs
    st.sidebar.title("Enter Your Health Information")
    
    # Data collection form
    with st.sidebar.form(key="health_form"):
        st.subheader("Demographics")
        sex = st.radio("Sex", options=["Male", "Female"])
        age_category = st.selectbox(
            "Age Category", 
            options=[
                "Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
                "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
                "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
                "Age 80 or older"
            ]
        )
        
        st.subheader("General Health")
        general_health = st.selectbox(
            "How would you rate your general health?", 
            options=["Excellent", "Very good", "Good", "Fair", "Poor"]
        )
        
        # Add to the Medical Conditions section in the form
        st.subheader("Medical Conditions")
        high_bp = st.selectbox("Do you have high blood pressure?", options=["Yes", "No"])
        high_chol = st.selectbox("Do you have high cholesterol?", options=["Yes", "No"])
        diabetes = st.selectbox("Do you have diabetes?", options=["Yes", "No"])
        stroke = st.selectbox("Have you ever had a stroke?", options=["Yes", "No"])
        angina = st.selectbox("Do you have angina/coronary heart disease?", options=["Yes", "No"])
        copd = st.selectbox("Do you have COPD?", options=["Yes", "No"])
        kidney = st.selectbox("Do you have kidney disease?", options=["Yes", "No"])

        # Add the missing fields
        vision_difficulty = st.selectbox("Do you have serious difficulty seeing, even with glasses?", options=["Yes", "No"])

        st.subheader("Lifestyle Factors")
        diff_walking = st.selectbox("Do you have difficulty walking?", options=["Yes", "No"])
        alcohol_drinker = st.selectbox("Have you had alcoholic beverages in the past 30 days?", options=["Yes", "No"])
        smoker_status = st.selectbox(
            "What is your smoking status?", 
            options=[
                "Never smoked", 
                "Former smoker",
                "Current smoker - now smokes some days", 
                "Current smoker - now smokes every day"
            ]
        )
        
        submit_button = st.form_submit_button(label="Predict Risk")
    
    # Main panel
    st.subheader("About this Dashboard")
    st.markdown("""
    This dashboard uses machine learning models trained on health survey data to predict 
    your risk of experiencing a heart attack. The prediction is based on a combination of:
    
    - Demographic information
    - Medical history
    - Lifestyle factors
    - General health status
    
    To get your risk assessment, fill out all the fields in the sidebar and click "Predict Risk".
    """)
    
    # Show technical information about the models
    with st.expander("Technical Information"):
        st.markdown("""
        ### Model Information
        - The prediction uses an ensemble of three machine learning models:
          - XGBoost
          - Random Forest
          - Logistic Regression
        
        - The models were trained on CDC's Behavioral Risk Factor Surveillance System (BRFSS) data
        
        - The final prediction is an average of these three models
        
        ### Risk Factors
        The key risk factors for heart attack include:
        - Advanced age
        - High blood pressure
        - High cholesterol
        - Diabetes
        - Previous stroke
        - COPD
        - Kidney disease
        - Smoking status
        - Difficulty walking
        - Poor general health
        """)
    
    # Create input data and make prediction
    if submit_button:
        # Create a DataFrame with user inputs
        input_data = pd.DataFrame({
            'Sex': [sex],
            'GeneralHealth': [general_health],
            'HighCholesterol': [high_chol],
            'HighBloodPressure': [high_bp],
            'HadAngina': [angina],
            'HadStroke': [stroke],
            'HadCOPD': [copd],
            'HadKidneyDisease': [kidney],
            'DifficultyWalking': [diff_walking],
            'SmokerStatus': [smoker_status],
            'AgeCategory': [age_category],
            'HadDiabetes': [diabetes],
            'BlindOrVisionDifficulty': [vision_difficulty],
            'AlcoholDrinkers': [alcohol_drinker]
        })
        
        try:
            # Initialize predictor
            predictor = HeartAttackPredictor()
            
            # Make prediction
            with st.spinner("Calculating your risk..."):
                result = predictor.predict(input_data)
            
            # Display risk assessment
            st.header("Your Heart Attack Risk Assessment")
            
            # Risk level with color coding
            risk_level = result['risk_level']
            risk_color = {
                'Low Risk': 'green',
                'Moderate Risk': 'orange',
                'High Risk': 'red',
                'Very High Risk': 'darkred'
            }[risk_level]
            
            # Display risk level
            st.markdown(f"### Risk Level: <span style='color:{risk_color};font-weight:bold'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Create columns for risk probability and risk factors
            col1, col2 = st.columns([1, 1])
            
            # Display risk probability
            with col1:
                st.subheader("Risk Probability")
                
                # Create gauge chart for risk probability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['probability'] * 100,
                    title={'text': "Risk Percentage"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 40], 'color': "lightyellow"},
                            {'range': [40, 60], 'color': "orange"},
                            {'range': [60, 100], 'color': "salmon"},
                        ]
                    }
                ))
                
                fig.update_layout(height=300, width=400)
                st.plotly_chart(fig)
            
            # Display risk factors
            with col2:
                st.subheader("Identified Risk Factors")
                if result['risk_factors']:
                    for factor in result['risk_factors']:
                        st.markdown(f"- {factor}")
                else:
                    st.write("No significant risk factors identified.")
            
            # Display model predictions
            st.subheader("Model Predictions")
            model_names = list(result['model_predictions'].keys())
            model_probs = [result['model_predictions'][model] * 100 for model in model_names]
            
            # Create bar chart for model predictions
            fig = px.bar(
                x=model_names,
                y=model_probs,
                labels={'x': 'Model', 'y': 'Probability (%)'},
                color=model_probs,
                color_continuous_scale=px.colors.sequential.Reds
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig)
            
            # Recommendations section
            st.header("Recommendations")
            
            if result['risk_level'] in ['High Risk', 'Very High Risk']:
                st.markdown("""
                ### Consult with a Healthcare Provider
                Based on your risk assessment, it is strongly recommended that you consult with a healthcare 
                provider to discuss your heart health and develop a personalized care plan.
                """)
            
            st.markdown("""
            ### General Heart Health Recommendations:
            1. *Regular Exercise*: Aim for at least 150 minutes of moderate-intensity exercise per week
            2. *Balanced Diet*: Focus on fruits, vegetables, whole grains, and lean proteins
            3. *Quit Smoking*: If you currently smoke, develop a plan to quit
            4. *Limit Alcohol*: If you drink alcohol, do so in moderation
            5. *Manage Stress*: Develop healthy coping strategies for stress
            6. *Regular Check-ups*: Schedule regular health check-ups with your doctor
            """)
            
            # Disclaimer
            st.info("*Disclaimer*: This tool provides an estimate of heart attack risk based on statistical models. It does not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 Heart Attack Risk Predictor")
    st.sidebar.markdown("Developed by: Rhendy & Sheyla")

if __name__ == "__main__":
    main()