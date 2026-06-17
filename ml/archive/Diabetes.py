# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from imblearn.over_sampling import SMOTE
from collections import Counter

plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset
print("Loading dataset...")
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'diabetes.csv')
df = pd.read_csv(dataset_path)

print(f"Dataset shape: {df.shape}")
print("\nClass distribution in raw dataset:")
print(df['Outcome'].value_counts())

# 2. Preprocessing: Handle missing/zero values
# Columns where 0 makes no sense physiologically and represents missing data
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 with NaN first
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)

# Impute NaN with median values grouped by Outcome to avoid data leakage bias
for col in zero_cols:
    df[col] = df.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))

# 3. Train-Test Split
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 4. Handle Class Imbalance using SMOTE on training data only
print("\nApplying SMOTE to balance classes in training data...")
print(f"Original training class distribution: {Counter(y_train)}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Balanced training class distribution: {Counter(y_train_res)}")

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use in production/inference
joblib.dump(scaler, os.path.join(script_dir, 'scaler.pkl'))
print("Scaler saved successfully as 'scaler.pkl'")

# 6. Model Training and Comparison
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gaussian NB': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = []
trained_models = {}

print("\nTraining and evaluating models...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train_res)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    trained_models[name] = model
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Convert results to DataFrame for nice visualization
results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print("\nModel Comparison Table:")
print(results_df.to_string(index=False))

# Select the best model (using F1-Score as the primary metric, since it balances precision and recall)
best_model_info = results_df.iloc[0]
best_model_name = best_model_info['Model']
best_model = trained_models[best_model_name]

print(f"\nBest Model selected: {best_model_name} with F1-Score: {best_model_info['F1-Score']:.4f}")

# Save the best model
model_filename = os.path.join(script_dir, 'best_diabetes_model.pkl')
joblib.dump(best_model, model_filename)
print(f"Best model saved successfully as '{os.path.basename(model_filename)}'")

# 7. Detailed evaluation of the best model
print(f"\nDetailed Classification Report for {best_model_name}:")
y_pred_best = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# Optional: Feature Importance for Tree-Based Models
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importances:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. {X.columns[indices[f]]}: {importances[indices[f]]:.4f}")

print("\nTraining workflow completed successfully!")
