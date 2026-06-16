# 🩺 Diabetes Prediction System

> **An end-to-end diabetes prediction system with Tkinter GUI, Flask Web Interface, and RESTful API.**

---

## 📌 Overview

Diabetes is one of the most prevalent chronic diseases worldwide. Early detection can significantly improve patient outcomes. This project provides a complete machine learning solution for diabetes risk assessment, accessible through:

- 🖥️ **Desktop Application** (Tkinter GUI)
- 🌐 **Web Interface** (Flask + HTML/CSS/JS)
- 🔌 **RESTful API** for integration with other systems

The system uses **7 different ML algorithms**, handles imbalanced data with **SMOTE**, and achieves high accuracy with a **Decision Tree** model.

---

## ✨ Features

### 🧠 Machine Learning
- ✅ **7 ML Models** compared (Logistic Regression, SVM, KNN, Decision Tree, Naive Bayes, Random Forest, Gradient Boosting)
- ✅ **SMOTE** (Synthetic Minority Oversampling Technique) for handling class imbalance
- ✅ **StandardScaler** for feature normalization
- ✅ **Feature Importance Analysis** for model interpretability
- ✅ **Model Export** (joblib) for production use

### 🖥️ Desktop Application (Tkinter)
- ✅ User-friendly interface with 8 input fields
- ✅ Instant prediction with confidence score
- ✅ Clear and reset functionality

### 🌐 Web Interface (Flask)
- ✅ Modern dark-themed responsive design
- ✅ Real-time prediction with animated confidence bar
- ✅ Keyboard support (Enter key)
- ✅ Mobile-friendly
- ✅ Professional error handling with toast notifications

### 🔌 RESTful API
- ✅ **POST** `/predict` endpoint
- ✅ JSON request/response format
- ✅ CORS ready for cross-origin requests
- ✅ Easy integration with other applications

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend API** | Flask 3.0.0 |
| **Machine Learning** | Scikit-learn 1.8.0 |
| **Data Processing** | Pandas, NumPy |
| **Data Balancing** | Imbalanced-learn (SMOTE) |
| **Visualization** | Matplotlib |
| **Desktop GUI** | Tkinter |
| **Web Frontend** | HTML5, CSS3, JavaScript |
| **Model Serialization** | Joblib |

---

## 📊 Model Performance

**Best Model:** Decision Tree Classifier

| Metric     | Score  |
|------------|--------|
| Accuracy   | 78.2%  |
| Precision  | 76.5%  |
| Recall     | 79.1%  |
| F1-Score   | 77.8%  |

### Feature Importance (Decision Tree)
