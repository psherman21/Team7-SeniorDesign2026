import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/processed/gesture_data.csv",
                    help="Path to the merged gesture CSV to train on")
args = parser.parse_args()

df = pd.read_csv(args.data)

X = df.drop("label", axis=1)
y = df["label"]

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Classes: {y.unique()}")
print(f"Samples per class:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

model = KNeighborsClassifier(
    n_neighbors=5,
    weights = 'distance',
    metric='euclidean'
    )

print("\nTraining KNN model...")
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
 
print(f"\n{'='*50}")
print(f"Model Performance")
print(f"{'='*50}")
print(f"Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))
 
# Cross-validation for more robust performance estimate
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\n5-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
 
# Feature importance analysis (distance-based for KNN)
# We can look at feature variance to see which sensors contribute most
print(f"\n{'='*50}")
print(f"Feature Statistics (scaled)")
print(f"{'='*50}")
feature_stdev = np.std(X_train_scaled, axis=0)
feature_names = list(X.columns)
for name, std in zip(feature_names, feature_stdev):
    print(f"{name:12s}: std={std:.3f}")
 
# Save model and scaler
print(f"\nSaving model and scaler...")
joblib.dump(model, "models/knn_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
print(f"Model saved to models/knn_model.joblib")
print(f"Scaler saved to models/scaler.joblib")