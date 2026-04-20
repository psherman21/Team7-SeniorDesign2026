import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/Ryan",
                    help="Path to a folder of gesture_*.csv files OR a single merged CSV")
parser.add_argument("--k", default=5, type=int,
                    help="Number of KNN neighbors (default 5)")
args = parser.parse_args()

# ── Load data — accepts either a folder or a single CSV ───────────────────────
data_path = Path(args.data)

if data_path.is_dir():
    csv_files = sorted(data_path.glob("gesture_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No gesture_*.csv files found in {data_path}")
    print(f"Loading {len(csv_files)} gesture file(s) from {data_path}:")
    frames = []
    for f in csv_files:
        df_f = pd.read_csv(f)
        print(f"  {f.name}: {len(df_f)} rows")
        frames.append(df_f)
    df = pd.concat(frames, ignore_index=True)
else:
    df = pd.read_csv(data_path)

# Drop all-zero columns (e.g. IMU not connected)
zero_cols = [c for c in df.columns if c != "label" and df[c].std() == 0 and df[c].mean() == 0]
if zero_cols:
    print(f"\nDropping zero-variance columns: {zero_cols}")
    df = df.drop(columns=zero_cols)

X = df.drop("label", axis=1)
y = df["label"]

print(f"\nDataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Classes: {sorted(y.unique())}")
print(f"\nSamples per class:")
print(y.value_counts().sort_index().to_string())

# ── Split + scale ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTraining set: {len(X_train)} | Test set: {len(X_test)}")

# ── Train ─────────────────────────────────────────────────────────────────────
model = KNeighborsClassifier(
    n_neighbors=args.k,
    weights="distance",
    metric="euclidean"
)
print(f"\nTraining KNN (k={args.k})...")
model.fit(X_train_scaled, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Model Performance")
print(f"{'='*50}")
print(f"Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(model, X_train_scaled, y_train,
                             cv=min(5, y_train.nunique()))
print(f"5-Fold CV: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*2*100:.1f}%)")

# ── Confusion Matrix figure ───────────────────────────────────────────────────
print(f"\nGenerating confusion matrix figure...")
labels = sorted(y.unique())
fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))

disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=labels,
    cmap="Blues",
    ax=ax
)
ax.set_title(f"KNN Gesture Recognition — Confusion Matrix\n"
             f"(k={args.k}, accuracy={acc*100:.1f}%, n={len(X_test)} test samples)",
             fontsize=13, pad=12)
plt.tight_layout()

out_path = Path("models/confusion_matrix.png")
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path.resolve()}")
plt.show()

# ── Feature statistics ────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("Feature Statistics (scaled)")
print(f"{'='*50}")
feature_stdev = np.std(X_train_scaled, axis=0)
for name, std in zip(X.columns, feature_stdev):
    print(f"  {name:12s}: std={std:.3f}")

# ── Save model ────────────────────────────────────────────────────────────────
print(f"\nSaving model and scaler...")
Path("models").mkdir(exist_ok=True)
joblib.dump(model,  "models/knn_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
print(f"Model saved  → models/knn_model.joblib")
print(f"Scaler saved → models/scaler.joblib")