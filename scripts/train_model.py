# Philip Sherman
# Team 7 - Senior Design
# 2/2/2026

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

df = pd.read_csv("data/processed/fake_gesture_data.csv") #change when actually running data collection

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/knn_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")