# Philip Sherman
# Team 7 - Senior Design
# 2/2/2025

import numpy as np
import pandas as pd

np.random.seed(42)

GESTURES = {
    "A": [80, 80, 80, 80, 80],
    "B": [10, 10, 10, 10, 10],
    "C": [40, 45, 50, 45, 40],
    "L": [10, 10, 80, 80, 80],
    "Y": [10, 80, 80, 80, 10],
}

SAMPLES_PER_GESTURE = 150

rows = []

for label, base in GESTURES.items():
    for _ in range(SAMPLES_PER_GESTURE):
        noise = np.random.normal(0, 5, size=5)
        sample = np.clip(np.array(base) + noise, 0, 100)
        rows.append([*sample, label])

df = pd.DataFrame(
    rows, columns=["thumb", "index", "middle", "ring", "pinky", "label"]
)

df.to_csv("data/processed/fake_gesture_data.csv", index=False)
print("Fake dataset generated.")