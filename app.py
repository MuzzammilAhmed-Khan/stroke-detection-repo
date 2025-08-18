from flask import Flask, render_template, request, url_for, session, redirect, flash, Response, redirect, jsonify
import numpy as np
import pandas as pd
import pickle
from xai_integration import StrokeXAIEngine
from enhanced_ordinal_classifier import EnhancedOrdinalStrokeClassifier

app = Flask(__name__)
app.secret_key = 'your secret key'

with open('enhanced_XGBstroke.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('enhanced_XGB_RFstroke.pkl', 'rb') as model2_file:
    model2 = pickle.load(model2_file)

# Load enhanced ordinal classifier
try:
    with open('model/enhanced_ordinal_classifier.pkl', 'rb') as ordinal_file:
        ordinal_classifier = pickle.load(ordinal_file)
    print("✅ Enhanced Ordinal Risk Classifier loaded successfully!")
    use_ordinal_classifier = True
except FileNotFoundError:
    try:
        with open('model/ordinal_stroke_classifier.pkl', 'rb') as ordinal_file:
            ordinal_classifier = pickle.load(ordinal_file)
        print("✅ Standard Ordinal Risk Classifier loaded successfully!")
        use_ordinal_classifier = True
    except FileNotFoundError:
        print("⚠️ No ordinal classifier found, using ensemble approach")
        ordinal_classifier = None
        use_ordinal_classifier = False

# Initialize XAI Engine
print("🚀 Initializing XAI Engine...")
xai_engine = StrokeXAIEngine()
xai_engine.initialize()
print("✅ XAI Engine ready for explanations!")

#======================================================================== ENHANCED RISK ASSESSMENT FUNCTIONS =============================================================================================================

def calculate_clinical_risk_score(input_dict):
    """
    Calculate clinical risk score based on medical risk factors
    Returns score 0-20+ (higher = more risk)
    """
    score = 0
    age = input_dict.get('age', 0)
    hypertension = input_dict.get('hypertension', 0)
    heart_disease = input_dict.get('heart_disease', 0)
    glucose = input_dict.get('avg_glucose_level', 0)
    bmi = input_dict.get('bmi', 0)
    smoking = input_dict.get('smoking_status', 0)
    
    # Age risk (0-4 points)
    if age >= 80: score += 4
    elif age >= 75: score += 3
    elif age >= 65: score += 2
    elif age >= 55: score += 1
    
    # Chronic conditions (0-4 points)
    if hypertension: score += 2
    if heart_disease: score += 2
    
    # Glucose levels (0-4 points)
    if glucose >= 300: score += 4
    elif glucose >= 250: score += 3
    elif glucose >= 200: score += 2
    elif glucose >= 140: score += 1
    
    # BMI (0-3 points)
    if bmi >= 40: score += 3
    elif bmi >= 35: score += 2
    elif bmi >= 30: score += 1
    
    # Smoking status (0-3 points)
    if smoking == 3: score += 3  # Current smoker
    elif smoking == 1: score += 2  # Former smoker
    elif smoking == 0: score += 1  # Unknown status
    
    return score

def determine_stroke_risk(ensemble_prob, clinical_score, input_dict):
    """
    Determine stroke risk using hybrid approach with ordinal classifier
    Returns comprehensive risk assessment
    """
    age = input_dict.get('age', 0)
    
    # Use ordinal classifier if available
    if use_ordinal_classifier and ordinal_classifier:
        try:
            # Prepare input for ordinal classifier
            feature_data = np.array([[
                input_dict.get('gender', 0),
                input_dict.get('age', 0),
                input_dict.get('hypertension', 0),
                input_dict.get('heart_disease', 0),
                input_dict.get('ever_married', 0),
                input_dict.get('work_type', 0),
                input_dict.get('Residence_type', 0),
                input_dict.get('avg_glucose_level', 0),
                input_dict.get('bmi', 0),
                input_dict.get('smoking_status', 0)
            ]])
            
            # Get ordinal risk prediction
            risk_probs = ordinal_classifier.predict_proba(feature_data)[0]
            risk_labels = ordinal_classifier.risk_labels
            predicted_risk = ordinal_classifier.predict_risk_labels(feature_data)[0]
            
            # Set alert levels based on ordinal prediction
            if predicted_risk == "EXTREME":
                alert_level = "STROKE DETECTED - EXTREME RISK"
                recommendation = "Immediate medical attention required"
            elif predicted_risk == "HIGH":
                alert_level = "STROKE DETECTED - HIGH RISK"
                recommendation = "Urgent medical evaluation recommended"
            elif predicted_risk == "ELEVATED":
                alert_level = "ELEVATED STROKE RISK"
                recommendation = "Medical consultation recommended"
            elif predicted_risk == "MODERATE":
                alert_level = "MODERATE STROKE RISK"
                recommendation = "Monitor risk factors closely"
            else:  # LOW
                alert_level = "LOW STROKE RISK"
                recommendation = "Continue healthy lifestyle practices"
            
            return {
                'alert_level': alert_level,
                'risk_category': predicted_risk,
                'ensemble_probability': ensemble_prob,
                'clinical_score': clinical_score,
                'recommendation': recommendation,
                'detection_method': 'Ordinal Classification',
                'ordinal_probabilities': {
                    'LOW': f"{risk_probs[0]:.3f}",
                    'MODERATE': f"{risk_probs[1]:.3f}",
                    'ELEVATED': f"{risk_probs[2]:.3f}",
                    'HIGH': f"{risk_probs[3]:.3f}",
                    'EXTREME': f"{risk_probs[4]:.3f}"
                }
            }
        except Exception as e:
            print(f"⚠️ Ordinal classifier error: {e}, falling back to ensemble method")
    
    # Fallback to original ensemble approach
    if ensemble_prob >= 0.999:  # Top 25% of stroke cases (very high confidence)
        alert_level = "STROKE DETECTED - EXTREME RISK"
        risk_category = "EXTREME"
        recommendation = "Immediate medical attention required"
    elif ensemble_prob >= 0.650:  # Median of stroke cases
        alert_level = "STROKE DETECTED - HIGH RISK"
        risk_category = "HIGH"
        recommendation = "Urgent medical evaluation recommended"
    elif ensemble_prob >= 0.278:  # Top 10% of no-stroke cases
        alert_level = "ELEVATED STROKE RISK"
        risk_category = "ELEVATED"
        recommendation = "Medical consultation recommended"
    elif ensemble_prob >= 0.100:  # Moderate threshold
        alert_level = "MODERATE STROKE RISK"
        risk_category = "MODERATE"
        recommendation = "Monitor risk factors closely"
    else:
        # Secondary clinical rule-based assessment for missed cases
        if clinical_score >= 12:
            alert_level = "ELEVATED STROKE RISK - CLINICAL ALERT"
            risk_category = "ELEVATED"
            recommendation = "Multiple risk factors present - medical evaluation recommended"
        elif clinical_score >= 8 and age >= 65:
            alert_level = "MODERATE STROKE RISK - CLINICAL ALERT"
            risk_category = "MODERATE"
            recommendation = "Age and risk factors warrant medical discussion"
        else:
            alert_level = "LOW STROKE RISK"
            risk_category = "LOW"
            recommendation = "Continue healthy lifestyle practices"
    
    return {
        'alert_level': alert_level,
        'risk_category': risk_category,
        'ensemble_probability': ensemble_prob,
        'clinical_score': clinical_score,
        'recommendation': recommendation,
        'detection_method': 'Model-based' if ensemble_prob >= 0.2 else 'Clinical rule-based'
    }

#======================================================================== HOME PAGE =============================================================================================================
@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html")

#=================================================================== LOGIN PAGE =====================================================================================================
@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == 'admin@gmail.com' and password == 'admin':
            msg = "Logged in successfully !"
            return render_template('upload.html', msg=msg)
        else:
            msg = "Invalid credentials or error"
            return render_template('login.html', msg=msg)
    return render_template('login.html')

#=================================================================== UPLOAD PAGE =====================================================================================================
@app.route("/upload")
def upload():
    return render_template("upload.html")

#=================================================================== PREVIEW PAGE =====================================================================================================
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        try:
            dataset = request.files['datasetfile']
            filename = dataset.filename.lower()
            
            # Handle different file formats
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(dataset)
            elif filename.endswith('.csv'):
                df = pd.read_csv(dataset, encoding='unicode_escape')
            else:
                return render_template("upload.html", error_msg="Unsupported file format. Please upload CSV or Excel files.")
            
            # Only set 'id' as index if it exists
            if 'id' in df.columns:
                df.set_index('id', inplace=True)
            
            # Limit display to first 100 rows for performance
            df_display = df.head(100) if len(df) > 100 else df
            
            return render_template("preview.html", 
                                 df_view=df_display, 
                                 total_rows=len(df),
                                 filename=dataset.filename)
        except Exception as e:
            print(f"Error in preview: {e}")
            return render_template("upload.html", error_msg=f"Error processing file: {str(e)}")	


#=================================================================== PREDICTION PAGE =====================================================================================================
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    return render_template("prediction.html")

@app.route("/predictions", methods=["GET", "POST"])
def predictions():
    if request.method == "POST":
        try:
            # Get form data
            gender = request.form["gender"]
            age = request.form["age"]
            hypertension = request.form["hypertension"]
            heart_disease = request.form["heart_disease"]
            ever_married = request.form["ever_married"]
            work_type = request.form["work_type"]
            Residence_type = request.form["Residence_type"]
            avg_glucose_level = request.form["avg_glucose_level"]
            bmi = request.form["bmi"]
            smoking_status = request.form["smoking_status"]

            # Create input dictionary first
            input_dict = {
                'gender': int(gender),
                'age': int(age),
                'hypertension': int(hypertension),
                'heart_disease': int(heart_disease),
                'ever_married': int(ever_married),
                'work_type': int(work_type),
                'Residence_type': int(Residence_type),
                'avg_glucose_level': float(avg_glucose_level),
                'bmi': float(bmi),
                'smoking_status': int(smoking_status)
            }

            # Prepare input data for model
            input_data = np.array([[int(gender), int(age), int(hypertension), int(heart_disease), 
                                    int(ever_married),  int(work_type), int(Residence_type), float(avg_glucose_level), 
                                    float(bmi), int(smoking_status) ]])
            print("input_data:", input_data)

            # Enhanced ensemble prediction approach
            # Get predictions from both models
            prob1 = model.predict_proba(input_data)[0]
            prob2 = model2.predict_proba(input_data)[0]
            
            # Use ensemble approach - take maximum risk assessment
            ensemble_stroke_prob = max(prob1[1], prob2[1])
            
            # Calculate clinical risk score
            clinical_risk_score = calculate_clinical_risk_score(input_dict)
            
            # Determine final prediction using hybrid approach
            prediction_result = determine_stroke_risk(ensemble_stroke_prob, clinical_risk_score, input_dict)
            
            result = prediction_result['alert_level']
            
            print("🔍 Generating explanations...")
            explanation = xai_engine.explain_prediction_for_web(input_dict)
            print("✅ Explanations generated successfully!")
            
            return render_template("prediction.html", 
                                 output=result,
                                 explanation=explanation,
                                 has_explanation=True,
                                 form_data=input_dict,
                                 risk_assessment=prediction_result,
                                 ensemble_info={
                                     'model1_prob': prob1[1],
                                     'model2_prob': prob2[1],
                                     'ensemble_prob': ensemble_stroke_prob
                                 })
        
        except Exception as e:
            print(f"❌ Error in prediction: {str(e)}")
            # Try to preserve form data even on error
            try:
                form_data = {
                    'gender': request.form.get('gender', ''),
                    'age': request.form.get('age', ''),
                    'hypertension': request.form.get('hypertension', ''),
                    'heart_disease': request.form.get('heart_disease', ''),
                    'ever_married': request.form.get('ever_married', ''),
                    'work_type': request.form.get('work_type', ''),
                    'Residence_type': request.form.get('Residence_type', ''),
                    'avg_glucose_level': request.form.get('avg_glucose_level', ''),
                    'bmi': request.form.get('bmi', ''),
                    'smoking_status': request.form.get('smoking_status', '')
                }
            except:
                form_data = {}
            
            return render_template("prediction.html", 
                                 output="Error in prediction",
                                 error_message=str(e),
                                 has_explanation=False,
                                 form_data=form_data)
            
    return render_template("prediction.html", has_explanation=False)

#=================================================================== XAI EXPLANATION API =====================================================================================================
@app.route("/api/explain", methods=["POST"])
def api_explain():
    """API endpoint for explanation generation"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        print(f"🔍 API explanation request: {data}")
        explanation = xai_engine.explain_prediction_for_web(data)
        print("✅ API explanation generated successfully!")
        
        return jsonify(explanation)
    
    except Exception as e:
        print(f"❌ API explanation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)