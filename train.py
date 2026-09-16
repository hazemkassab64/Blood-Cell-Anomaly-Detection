import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("blood_cell_anomaly_detection.csv")

df.drop('cell_id', axis=1, inplace=True)
df['patient_age_group'] = df['patient_age_group'].map({'Pediatric':0, 'Adult':1, 'Elderly':2})
df['patient_sex'] = LabelEncoder().fit_transform(df['patient_sex'])
df = pd.get_dummies(df, columns=['dataset_source', 'staining_protocol', 'microscope_model'], drop_first=True, dtype=int)

leakage_cols = [
    'cell_type',
    'disease_category',
    'cytodiffusion_anomaly_score',
    'cytodiffusion_classification_confidence',
    'labeller_confidence_score'
]

X_binary = df.drop(columns=leakage_cols + ['anomaly_label'], axis=1)
y_binary = df['anomaly_label']

X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

print("XGBoost Binary Classification Model Training...")
# BEST MODEL: XGBoost Classifier
xgb_binary = XGBClassifier()
xgb_binary.fit(X_train_binary, y_train_binary)


y_pred_binary_xgb = xgb_binary.predict(X_test_binary)
print(f"\nAccuracy: {accuracy_score(y_test_binary, y_pred_binary_xgb)}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary_xgb))

print("Training completed! Saving the model and columns for later use in the Streamlit app...")
joblib.dump(xgb_binary, 'xgb_model.pkl')
joblib.dump(X_train_binary.columns.tolist(), 'model_columns.pkl')
print("Model and columns saved successfully!")