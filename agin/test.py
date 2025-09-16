import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

# --- Model Definition ---
class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# --- Load Model + Scaler ---
scaler = joblib.load("scaler.pkl")
model = MLPModel(5, 5)
model.load_state_dict(torch.load("D:\\bike_share\\agin\\bike_fault_predictor.pkl", map_location=torch.device("cpu")))
model.eval()

# --- Generate Sample Bike Data for Dashboard ---
def generate_bike_data(n=20):
    bikes = []
    locations = ["Downtown", "Campus", "Riverfront", "Park West", "Hillside", 
                 "Market District", "Uptown", "Waterfront", "East End", "The Meadows"]
    
    for i in range(n):
        bike_id = f"BK{i+1:03d}"
        location = random.choice(locations)
        last_maintenance = datetime.now() - timedelta(days=random.randint(1, 180))
        
        # Generate sensor readings
        temperature = random.randint(40, 100)
        vibration = random.randint(2, 15)
        pressure = random.randint(80, 120)
        anomaly_score = round(random.uniform(0.1, 0.9), 2)
        ride_duration = random.randint(30, 240)
        
        # Create input for prediction
        X = np.array([[temperature, vibration, pressure, anomaly_score, ride_duration]])
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Get predictions
        with torch.no_grad():
            probs = model(X_tensor).numpy()[0]
        
        bikes.append({
            "id": bike_id,
            "location": location,
            "last_maintenance": last_maintenance.strftime("%Y-%m-%d"),
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "anomaly_score": anomaly_score,
            "ride_duration": ride_duration,
            "brakes_risk": probs[0],
            "tires_risk": probs[1],
            "chain_risk": probs[2],
            "gears_risk": probs[3],
            "electronics_risk": probs[4],
            "overall_risk": max(probs)  # Highest component risk
        })
    
    return bikes

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🚲 e-Bike Maintenance Dashboard")

# Sidebar for single bike prediction
with st.sidebar:
    st.header("🔧 Single Bike Assessment")
    
    st.subheader("📊 Why these features?")
    st.markdown("""
    We selected these **key sensor features** because they directly affect e-Bike health:
    - **Vibration (g-force):** Excessive vibration signals mechanical issues like loose parts or imbalanced wheels.
    - **Motor Temperature (°C):** High temperatures accelerate wear and can trigger sudden motor failures.
    - **Ride Duration (mins):** Longer rides increase cumulative stress, raising the chance of overheating and component fatigue.
    """)
    
    temperature = st.slider("Temperature (°C)", 0, 120, 70)
    vibration = st.slider("Vibration (mm/s)", 0, 20, 5)
    pressure = st.slider("Tire Pressure (psi)", 50, 200, 100)
    anomaly_score = st.slider("Anomaly Score", 0.0, 1.0, 0.5)
    ride_duration = st.slider("Ride Duration (mins)", 0, 300, 60)
    
    if st.button("🔮 Predict Failure Risks"):
        # Prepare input
        X = np.array([[temperature, vibration, pressure, anomaly_score, ride_duration]])
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Get predictions
        with torch.no_grad():
            probs = model(X_tensor).numpy()[0]
        
        parts = ["Brakes", "Tires", "Chain", "Gears", "Electronics"]
        
        st.subheader("Prediction Results")
        for p, prob in zip(parts, probs):
            st.write(f"**{p} Failure Risk:** {prob*100:.1f}%")
        
        # Show Bar Graph
        st.subheader("📊 Risk Visualization")
        fig, ax = plt.subplots()
        ax.bar(parts, probs * 100, color="orange")
        ax.set_ylabel("Failure Probability (%)")
        ax.set_ylim(0, 100)
        st.pyplot(fig)

# Main dashboard
st.header("🏍️ Fleet Overview")
bike_data = generate_bike_data()

# Display metrics
col1, col2, col3, col4 = st.columns(4)
high_risk_bikes = sum(1 for bike in bike_data if bike["overall_risk"] > 0.7)
col1.metric("Total Bikes", len(bike_data))
col2.metric("High Risk Bikes", high_risk_bikes)
col3.metric("Avg. Temperature", f"{sum(b['temperature'] for b in bike_data)/len(bike_data):.1f}°C")
col4.metric("Avg. Risk", f"{sum(b['overall_risk'] for b in bike_data)/len(bike_data)*100:.1f}%")

# Display bike table
st.subheader("📋 Bike Status")
df = pd.DataFrame(bike_data)
# Format risk columns as percentages
for col in ["brakes_risk", "tires_risk", "chain_risk", "gears_risk", "electronics_risk", "overall_risk"]:
    df[col] = df[col].apply(lambda x: f"{x*100:.1f}%")

st.dataframe(df)

# Display explanations for high-risk bikes
st.subheader("🔍 Component Risk Explanations")
for bike in bike_data:
    if bike["overall_risk"] > 0.7:
        issues = []
        if bike["brakes_risk"] > 0.7:
            issues.append("High vibration and temperature suggest brake wear")
        if bike["tires_risk"] > 0.7:
            issues.append("Abnormal pressure readings indicate potential tire issues")
        if bike["chain_risk"] > 0.7:
            issues.append("Elevated vibration suggests chain may need lubrication/replacement")
        if bike["gears_risk"] > 0.7:
            issues.append("Vibration patterns indicate potential gear alignment issues")
        if bike["electronics_risk"] > 0.7:
            issues.append("High temperature and anomaly score suggest electrical system concerns")
        
        if issues:
            st.write(f"**{bike['id']}** ({bike['location']}):")
            for issue in issues:
                st.write(f"- {issue}")

# Simple map visualization
st.subheader("🗺️ Bike Locations")
location_counts = {}
for bike in bike_data:
    location = bike["location"]
    location_counts[location] = location_counts.get(location, 0) + 1

fig, ax = plt.subplots()
ax.bar(location_counts.keys(), location_counts.values())
ax.set_ylabel("Number of Bikes")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Maintenance recommendations
st.subheader("🛠️ Maintenance Recommendations")
high_risk_bikes = [b for b in bike_data if b["overall_risk"] > 0.7]
if high_risk_bikes:
    st.write("The following bikes require immediate attention:")
    for bike in high_risk_bikes:
        st.write(f"- **{bike['id']}** at {bike['location']} (Last maintenance: {bike['last_maintenance']})")
else:
    st.write("No bikes currently require immediate maintenance. Schedule routine checks as needed.")