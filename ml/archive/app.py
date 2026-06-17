import tkinter as tk
from tkinter import ttk, messagebox
import os
import joblib
import pandas as pd
import numpy as np

# ===== Load Model & Scaler =====
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best_diabetes_model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Error", "Model files not found!\nPlease run 'Diabetes.py' first to train the model.")
    exit(1)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


# ===== Main Window =====
app = tk.Tk()
app.title("🩺 Diabetes Prediction System")
app.geometry("520x680")
app.resizable(False, False)
app.configure(bg="#f0f4f8")

# ===== Title =====
title_frame = tk.Frame(app, bg="#2563eb", pady=15)
title_frame.pack(fill="x")

tk.Label(
    title_frame, text="🩺 Diabetes Prediction",
    font=("Segoe UI", 20, "bold"), fg="white", bg="#2563eb"
).pack()
tk.Label(
    title_frame, text="Enter patient details below",
    font=("Segoe UI", 11), fg="#bfdbfe", bg="#2563eb"
).pack()


# ===== Input Frame =====
input_frame = tk.Frame(app, bg="#f0f4f8", padx=30, pady=20)
input_frame.pack(fill="x")

# Feature configs: (label, placeholder, default)
features = [
    ("Pregnancies", "e.g. 2", "2"),
    ("Glucose Level", "e.g. 120", "120"),
    ("Blood Pressure", "e.g. 70", "70"),
    ("Skin Thickness", "e.g. 20", "20"),
    ("Insulin Level", "e.g. 80", "80"),
    ("BMI", "e.g. 25.5", "25.5"),
    ("Diabetes Pedigree Func", "e.g. 0.35", "0.35"),
    ("Age", "e.g. 30", "30"),
]

entries = []

for i, (label_text, placeholder, default) in enumerate(features):
    row = i // 2
    col = (i % 2) * 2

    lbl = tk.Label(
        input_frame, text=label_text + ":",
        font=("Segoe UI", 10, "bold"), bg="#f0f4f8", fg="#1e293b",
        anchor="w"
    )
    lbl.grid(row=row, column=col, columnspan=2, sticky="w", padx=(10, 0), pady=(8, 0))

    entry = tk.Entry(
        input_frame, font=("Segoe UI", 11), width=16,
        relief="solid", bd=1, bg="white", fg="#334155"
    )
    entry.grid(row=row + 1, column=col, columnspan=2, sticky="w", padx=(10, 0), pady=(2, 0))
    entry.insert(0, default)
    entries.append(entry)

# Add some vertical space
spacer_row = len(features) // 2 + 2
tk.Frame(input_frame, height=5, bg="#f0f4f8").grid(row=spacer_row, column=0)


# ===== Buttons Frame =====
btn_frame = tk.Frame(app, bg="#f0f4f8")
btn_frame.pack(pady=5)

def clear_fields():
    for entry in entries:
        entry.delete(0, tk.END)

def predict():
    try:
        values = [float(e.get()) for e in entries]
    except ValueError:
        messagebox.showwarning("Invalid Input", "Please enter valid numbers in all fields!")
        return

    input_data = pd.DataFrame([{
        'Pregnancies': values[0],
        'Glucose': values[1],
        'BloodPressure': values[2],
        'SkinThickness': values[3],
        'Insulin': values[4],
        'BMI': values[5],
        'DiabetesPedigreeFunction': values[6],
        'Age': values[7]
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None

    # Show result
    show_result(prediction, prediction_proba)

def show_result(prediction, proba):
    result_win = tk.Toplevel(app)
    result_win.title("Result")
    result_win.geometry("400x280")
    result_win.resizable(False, False)
    result_win.configure(bg="white")
    result_win.grab_set()

    if prediction == 1:
        color = "#dc2626"
        emoji = "🔴"
        text = "DIABETES"
        detail = "The model predicts that this patient HAS diabetes."
        conf = f"Confidence: {proba[1]*100:.1f}%" if proba is not None else ""
    else:
        color = "#16a34a"
        emoji = "🟢"
        text = "NO DIABETES"
        detail = "The model predicts that this patient is HEALTHY."
        conf = f"Confidence: {proba[0]*100:.1f}%" if proba is not None else ""

    tk.Label(result_win, text=emoji, font=("Segoe UI", 50), bg="white").pack(pady=(20, 5))
    tk.Label(result_win, text=text, font=("Segoe UI", 22, "bold"),
             fg=color, bg="white").pack()
    tk.Label(result_win, text=detail, font=("Segoe UI", 11),
             fg="#475569", bg="white").pack(pady=(5, 0))
    if conf:
        tk.Label(result_win, text=conf, font=("Segoe UI", 12, "bold"),
                 fg="#1e293b", bg="white").pack(pady=(8, 0))

    tk.Button(
        result_win, text="Close", font=("Segoe UI", 11, "bold"),
        bg="#e2e8f0", fg="#1e293b", relief="flat", padx=30, pady=5,
        command=result_win.destroy, cursor="hand2"
    ).pack(pady=(15, 10))


# Predict Button
predict_btn = tk.Button(
    btn_frame, text="🔍  Predict", font=("Segoe UI", 13, "bold"),
    bg="#2563eb", fg="white", relief="flat", padx=40, pady=8,
    command=predict, cursor="hand2", activebackground="#1d4ed8"
)
predict_btn.grid(row=0, column=0, padx=10)

# Clear Button
clear_btn = tk.Button(
    btn_frame, text="🗑  Clear", font=("Segoe UI", 13, "bold"),
    bg="#e2e8f0", fg="#475569", relief="flat", padx=40, pady=8,
    command=clear_fields, cursor="hand2", activebackground="#cbd5e1"
)
clear_btn.grid(row=0, column=1, padx=10)


# ===== Footer =====
footer = tk.Frame(app, bg="#2563eb", pady=8)
footer.pack(side="bottom", fill="x")
tk.Label(
    footer, text="Powered by Machine Learning  |  Diabetes Prediction System",
    font=("Segoe UI", 9), fg="#bfdbfe", bg="#2563eb"
).pack()


# ===== Run =====
app.mainloop()