import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Create/Load Dataset (Using a robust synthetic generator matching clinical parameters with 14 features & 1000 instances)
X_raw, y_raw = make_classification(
    n_samples=1000, 
    n_features=14, 
    n_informative=10, 
    n_redundant=4, 
    random_state=42
)

feature_names = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 
    'Oldpeak', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_ASY',
    'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up'
]

df = pd.DataFrame(X_raw, columns=feature_names)
# Make some features strictly binary/integer-like for realism
df['FastingBS'] = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
df['ExerciseAngina_Y'] = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])
df['Target'] = y_raw

# Split into Train and Test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save test data for Streamlit app
os.makedirs("model", exist_ok=True)
test_df.to_csv("test_data.csv", index=False)

X_train = train_df.drop(columns=['Target'])
y_train = train_df['Target']
X_test = test_df.drop(columns=['Target'])
y_test = test_df['Target']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/scaler.pkl")

# Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

results = []

for name, model in models.items():
    # Train
    if name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Save model
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.pkl")
    
    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })

results_df = pd.DataFrame(results)
print("Evaluation Metrics Summary:")
print(results_df.to_string(index=False))