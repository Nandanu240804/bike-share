# train_multilabel_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib

# -----------------------------
# 1. Generate synthetic dataset
# -----------------------------
np.random.seed(42)
N = 2000

data = pd.DataFrame({
    "vibration": np.random.uniform(0, 5, N),
    "motor_temp": np.random.uniform(20, 120, N),
    "ride_duration": np.random.randint(5, 180, N),
    "avg_speed": np.random.uniform(5, 40, N),
    "braking_intensity": np.random.uniform(0, 10, N),
    "terrain_slope": np.random.uniform(0, 15, N),
})

# Define rules for failures
data["brakes_fail"] = ((data["braking_intensity"] > 7) & (data["ride_duration"] > 90)).astype(int)
data["tires_fail"] = ((data["vibration"] > 3) & (data["terrain_slope"] > 8)).astype(int)
data["chain_fail"] = ((data["ride_duration"] > 120) & (data["motor_temp"] > 80)).astype(int)

X = data.drop(columns=["brakes_fail", "tires_fail", "chain_fail"])
y = data[["brakes_fail", "tires_fail", "chain_fail"]]

# -----------------------------
# 2. Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
model.fit(X_train, y_train)

# -----------------------------
# 3. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Brakes", "Tires", "Chain"]))

# -----------------------------
# 4. Save model
# -----------------------------
joblib.dump(model, "bike_fault_predictor.pkl")
print("✅ Model saved as bike_fault_predictor.pkl")
