"""
Complete XAI Integration for Stroke Prediction Web Application
Combines SHAP, LIME, and visualization capabilities
"""
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, Tuple, Optional
from explainer_setup import StrokeExplainerManager
from explanation_visualizer import ExplanationVisualizer
from data_preprocessing import load_training_data
from medical_context_explainer import get_medical_interpretation, create_medical_summary

class StrokeXAIEngine:
    """Complete XAI engine for stroke prediction explanations"""
    
    def __init__(self):
        self.explainer_manager = None
        self.visualizer = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the complete XAI system"""
        print("🚀 Initializing Stroke XAI Engine...")
        
        try:
            # Initialize explainer manager
            self.explainer_manager = StrokeExplainerManager()
            success = self.explainer_manager.initialize_all_explainers()
            
            if not success:
                print("❌ Failed to initialize explainers")
                return False
            
            # Initialize visualizer
            self.visualizer = ExplanationVisualizer()
            
            self.is_initialized = True
            print("✅ XAI Engine initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing XAI Engine: {e}")
            return False
    
    def generate_complete_explanation(self, input_data: np.ndarray, 
                                    model_name: str = 'primary') -> Dict[str, Any]:
        """
        Generate complete explanation for a prediction
        
        Args:
            input_data: Input features for prediction [1 x 10 array]
            model_name: Which model to use ('primary' or 'secondary')
            
        Returns:
            Dictionary with all explanation components
        """
        if not self.is_initialized:
            raise RuntimeError("XAI Engine not initialized. Call initialize() first.")
        
        try:
            print(f"Generating explanation for {model_name} model...")
            
            # Step 1: Make prediction
            model = self.explainer_manager.models[model_name]
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Step 2: Generate SHAP explanation
            shap_explainer = self.explainer_manager.shap_explainers[model_name]
            shap_values = shap_explainer.shap_values(input_data)
            expected_value = shap_explainer.expected_value
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # For binary classification, take the positive class
                shap_values_processed = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                expected_value_processed = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            else:
                shap_values_processed = shap_values
                expected_value_processed = expected_value
            
            # Step 3: Generate LIME explanation
            lime_explanation = self.explainer_manager.lime_explainer.explain_instance(
                input_data.flatten(),
                model.predict_proba,
                num_features=len(self.visualizer.feature_names),
                top_labels=2
            )
            
            # Step 4: Generate visualizations
            visualizations = {}
            
            # SHAP bar plot
            visualizations['shap_bar_plot'] = self.visualizer.create_shap_bar_plot(shap_values_processed)
            
            # SHAP waterfall plot
            visualizations['shap_waterfall'] = self.visualizer.create_shap_waterfall_plot(
                shap_values_processed, expected_value_processed, input_data, prediction_proba[1]
            )
            
            # LIME plot
            visualizations['lime_plot'] = self.visualizer.create_lime_explanation_plot(lime_explanation)
            
            # Interactive feature plot
            visualizations['interactive_plot'] = self.visualizer.create_interactive_feature_plot(
                shap_values_processed, input_data
            )
            
            # Confidence gauge
            visualizations['confidence_gauge'] = self.visualizer.create_confidence_gauge(prediction_proba[1])
            
            # Step 5: Generate explanation summary
            explanation_summary = self.visualizer.generate_explanation_summary(
                shap_values_processed, lime_explanation, prediction_proba[1], input_data
            )
            
            # Step 6: Compile complete result
            result = {
                'prediction': {
                    'class': int(prediction),
                    'probability': prediction_proba.tolist(),
                    'stroke_probability': float(prediction_proba[1]),
                    'result_text': 'Stroke Detected' if prediction == 1 else 'No Stroke'
                },
                'shap_explanation': {
                    'values': shap_values_processed.tolist() if isinstance(shap_values_processed, np.ndarray) else shap_values_processed,
                    'expected_value': float(expected_value_processed),
                    'feature_names': self.visualizer.feature_names
                },
                'lime_explanation': {
                    'features': lime_explanation.as_list() if lime_explanation else [],
                    'score': lime_explanation.score if lime_explanation else 0
                },
                'visualizations': visualizations,
                'explanation_summary': explanation_summary,
                'input_data': {
                    'features': input_data.tolist(),
                    'feature_names': self.visualizer.feature_names,
                    'feature_descriptions': self.visualizer.feature_descriptions
                }
            }
            
            print("✅ Complete explanation generated successfully!")
            return result
            
        except Exception as e:
            print(f"❌ Error generating explanation: {e}")
            raise
    
    def explain_prediction_for_web(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation formatted for web application display
        
        Args:
            input_features: Dictionary with feature names and values
            
        Returns:
            Web-ready explanation dictionary
        """
        try:
            # Convert input features to numpy array in correct order
            feature_order = ['gender', 'age', 'hypertension', 'heart_disease', 
                           'ever_married', 'work_type', 'Residence_type', 
                           'avg_glucose_level', 'bmi', 'smoking_status']
            
            input_array = np.array([[
                float(input_features.get(feature, 0)) for feature in feature_order
            ]])
            
            # Generate complete explanation
            explanation = self.generate_complete_explanation(input_array)
            
            # Generate medical context interpretations
            medical_summary = create_medical_summary(
                explanation['explanation_summary']['top_risk_factors'],
                explanation['explanation_summary']['top_protective_factors'],
                input_features
            )
            
            # Format for web display
            web_explanation = {
                'prediction_result': explanation['prediction']['result_text'],
                'stroke_probability': explanation['prediction']['stroke_probability'],
                'confidence': explanation['prediction']['stroke_probability'],  # Add confidence field
                'confidence_percentage': explanation['prediction']['stroke_probability'] * 100,
                'risk_level': explanation['explanation_summary']['risk_level'],
                
                # Top factors with medical context
                'top_risk_factors': explanation['explanation_summary']['top_risk_factors'],
                'top_protective_factors': explanation['explanation_summary']['top_protective_factors'],
                
                # Medical context interpretations
                'medical_context': medical_summary,
                
                # Visualizations formatted for web
                'visualizations': [
                    {
                        'title': 'SHAP Feature Importance',
                        'data': explanation['visualizations']['shap_bar_plot'],
                        'format': 'base64',
                        'type': 'shap_bar'
                    },
                    {
                        'title': 'Risk Assessment',
                        'data': explanation['visualizations']['confidence_gauge'],
                        'format': 'base64',
                        'type': 'gauge'
                    }
                ] if explanation['visualizations'] else [],
                
                # Technical details for advanced users
                'technical_details': {
                    'shap_values': explanation['shap_explanation']['values'],
                    'lime_features': explanation['lime_explanation']['features'],
                    'model_expected_value': explanation['shap_explanation']['expected_value']
                }
            }
            
            return web_explanation
            
        except Exception as e:
            print(f"❌ Error generating web explanation: {e}")
            raise

def test_complete_integration():
    """Test the complete XAI integration"""
    print("🧪 Testing Complete XAI Integration...")
    print("=" * 60)
    
    # Initialize XAI engine
    engine = StrokeXAIEngine()
    success = engine.initialize()
    
    if not success:
        print("💥 Integration test failed - could not initialize engine")
        return
    
    # Test with sample data (high-risk patient)
    test_input = {
        'gender': 1,  # Male
        'age': 67,    # Elderly
        'hypertension': 1,  # Has hypertension
        'heart_disease': 1,  # Has heart disease
        'ever_married': 1,   # Married
        'work_type': 2,      # Private job
        'Residence_type': 1, # Urban
        'avg_glucose_level': 228.69,  # High glucose
        'bmi': 36.6,         # High BMI
        'smoking_status': 1  # Formerly smoked
    }
    
    print("Testing with high-risk patient profile...")
    try:
        web_explanation = engine.explain_prediction_for_web(test_input)
        
        print(f"✅ Prediction: {web_explanation['prediction_result']}")
        print(f"✅ Stroke Probability: {web_explanation['stroke_probability']:.3f}")
        print(f"✅ Risk Level: {web_explanation['risk_level']}")
        print(f"✅ Number of risk factors: {len(web_explanation['top_risk_factors'])}")
        print(f"✅ Number of protective factors: {len(web_explanation['top_protective_factors'])}")
        print(f"✅ Visualizations generated: {len([v for v in web_explanation.values() if 'plot' in str(v) or 'gauge' in str(v) if v])}")
        
        print("\\n🎯 Top Risk Factors:")
        for factor in web_explanation['top_risk_factors'][:3]:
            print(f"  • {factor['feature']}: {factor['shap_value']:.4f}")
        
        print("\\n🛡️ Top Protective Factors:")
        for factor in web_explanation['top_protective_factors'][:3]:
            print(f"  • {factor['feature']}: {factor['shap_value']:.4f}")
        
        print("\\n✅ Complete XAI integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    test_complete_integration()