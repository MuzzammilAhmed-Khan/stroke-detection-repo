# 🏥 Stroke Detection Application

An AI-powered web application for stroke risk prediction with explainable AI features.

## Features

- **Individual Risk Assessment**: Predict stroke risk for individual patients
- **Explainable AI**: SHAP and LIME explanations for predictions
- **5-Level Risk Classification**: LOW, MODERATE, ELEVATED, HIGH, EXTREME
- **Bulk Processing**: Upload CSV/Excel files for multiple predictions
- **Clinical Risk Scoring**: Combined AI and clinical rule-based assessment
- **Web Interface**: User-friendly web application

## Quick Start

See [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md) for complete setup and usage instructions.

### Basic Setup
```bash
# Create virtual environment
python3 -m venv stroke_prediction_env
source stroke_prediction_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

## Login Credentials
- Username: `admin@gmail.com`
- Password: `admin`

## Project Structure

- `app.py` - Main Flask application
- `xai_integration.py` - Explainable AI engine
- `enhanced_ordinal_classifier.py` - Risk classification model
- `templates/` - HTML templates
- `static/` - CSS, JS, and images
- `model/` - Machine learning models and data
- `*.pkl` - Trained model files

## Requirements

- Python 3.8+
- Flask
- scikit-learn
- XGBoost
- pandas
- numpy
- SHAP
- LIME

## License

This project is for educational and research purposes.