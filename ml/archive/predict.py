import os
import joblib
import pandas as pd
import numpy as np

# Load the saved model and scaler
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best_diabetes_model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print("Error: Model or Scaler not found! Please run 'Diabetes.py' first to train the model.")
    exit(1)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("=" * 60)
print("             Diabetes Prediction System (Test Tool)            ")
print("=" * 60)
print("Please enter the patient details below:")

def get_input(prompt, default_val=None):
    while True:
        try:
            val = input(prompt)
            if val.strip() == "" and default_val is not None:
                return default_val
            return float(val)
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

# Get feature inputs from the user
pregnancies = get_input("1. Number of Pregnancies (e.g., 2): ", 2.0)
glucose = get_input("2. Glucose Level (e.g., 120): ", 120.0)
bp = get_input("3. Blood Pressure (e.g., 70): ", 70.0)
thickness = get_input("4. Skin Thickness (e.g., 20): ", 20.0)
insulin = get_input("5. Insulin Level (e.g., 80): ", 80.0)
bmi = get_input("6. BMI (e.g., 25.5): ", 25.5)
dpf = get_input("7. Diabetes Pedigree Function (e.g., 0.35): ", 0.35)
age = get_input("8. Age (e.g., 30): ", 30.0)

# Create input DataFrame
input_data = pd.DataFrame([{
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': bp,
    'SkinThickness': thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}])

# Scale the inputs
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None

print("\n" + "="*60)
print("                           RESULTS                          ")
print("="*60)
if prediction == 1:
    print("🔴 Result: The model predicts DIABETES.")
    if prediction_proba is not None:
         print(f"Confidence Level: {prediction_proba[1]*100:.2f}%")
else:
    print("🟢 Result: The model predicts NO DIABETES (Healthy).")
    if prediction_proba is not None:
         print(f"Confidence Level: {prediction_proba[0]*100:.2f}%")
print("="*60)
