from flask import Flask, request, jsonify, send_from_directory
import os
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='static')

# ===== Load Model & Scaler =====
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best_diabetes_model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_data = pd.DataFrame([{
            'Pregnancies':               float(data['pregnancies']),
            'Glucose':                   float(data['glucose']),
            'BloodPressure':             float(data['bloodPressure']),
            'SkinThickness':             float(data['skinThickness']),
            'Insulin':                   float(data['insulin']),
            'BMI':                       float(data['bmi']),
            'DiabetesPedigreeFunction':  float(data['dpf']),
            'Age':                       float(data['age']),
        }])

        input_scaled = scaler.transform(input_data)
        prediction = int(model.predict(input_scaled)[0])

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            confidence = float(proba[1] if prediction == 1 else proba[0]) * 100
        else:
            confidence = None

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 1) if confidence is not None else None,
            'label': 'DIABETES' if prediction == 1 else 'NO DIABETES'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
