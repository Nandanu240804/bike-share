import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="e-Bike Maintenance Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)
import streamlit as st

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Critical Alert Card (C8) */
    .alert-critical {
        background-color: #ffe5e5; /* light red background */
        border-left: 6px solid #ff0000; /* red accent border */
        color: #2c2c2c; /* dark text */
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .alert-critical h3 {
        color: #b30000;
        font-weight: bold;
    }

    .alert-critical strong {
        color: #000;
    }

    .stButton>button {
        background-color: #d32f2f;
        color: white;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 6px;
        border: none;
        margin-top: 10px;
    }

    .stButton>button:hover {
        background-color: #b71c1c;
        transform: translateY(-1px);
        transition: 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<h1 class='main-header'>🚨 Critical Alerts</h1>", unsafe_allow_html=True)

# =========================
# Example Critical Alerts
# =========================
alerts = [
    {"id": "BK001", "location": "Waterfront", "component": "Brakes", "risk": "89.3%", "temp": "93°C"},
    {"id": "BK005", "location": "Uptown", "component": "Brakes", "risk": "82.1%", "temp": "85°C"},
    {"id": "BK008", "location": "Market District", "component": "Brakes", "risk": "88.4%", "temp": "70°C"},
    {"id": "BK009", "location": "Downtown", "component": "Brakes", "risk": "89.7%", "temp": "96°C"},
]

# =========================
# Render Alerts
# =========================
for alert in alerts:
    with st.container():
        st.markdown(
            f"""
            <div class="alert-critical">
                <h3>CRITICAL: {alert['id']} at {alert['location']}</h3>
                <p><strong>Component:</strong> {alert['component']}</p>
                <p><strong>Risk:</strong> {alert['risk']} &nbsp;&nbsp; <strong>Temp:</strong> {alert['temp']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(f"Schedule {alert['id']}"):
            st.success(f"✅ Maintenance scheduled for {alert['id']}")



# =========================
# Model Definition
# =========================
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

# =========================
# Try to Load Model/Scaler
# =========================
def try_load_model():
    try:
        # Try local folder first
        scaler = joblib.load("scaler.pkl")
        model = MLPModel(5, 5)
        model.load_state_dict(torch.load("bike_model.pth", map_location=torch.device("cpu")))
        model.eval()
        return scaler, model, True
    except Exception:
        # Try example Windows path (optional)
        try:
            scaler = joblib.load(r"D:\bike_share\scaler.pkl")
            model = MLPModel(5, 5)
            model.load_state_dict(torch.load("bike_model.pth", map_location=torch.device("cpu")))
            model.eval()
            return scaler, model, True
        except Exception:
            return None, None, False

if "model_loaded" not in st.session_state:
    st.session_state.scaler, st.session_state.model, st.session_state.model_loaded = try_load_model()

# =========================
# Route Optimizer
# =========================
class MaintenanceRouteOptimizer:
    def __init__(self):
        self.locations = {
            "Downtown": (2, 5),
            "Campus": (8, 3),
            "Riverfront": (1, 2),
            "Park West": (4, 7),
            "Hillside": (6, 9),
            "Market District": (3, 4),
            "Uptown": (7, 6),
            "Waterfront": (0, 1),
            "East End": (9, 2),
            "The Meadows": (5, 8)
        }
        self.maintenance_center = (5, 5)

    def distance(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def calculate_route_time(self, route):
        total = 0
        current = self.maintenance_center
        for loc in route:
            coords = self.maintenance_center if loc == "Maintenance Center" else self.locations[loc]
            total += self.distance(current, coords) * 3  # 3 min per unit distance
            current = coords
        return round(total)

    def optimize_routes(self, high_priority, medium_priority, max_bikes_per_route=5):
        all_bikes = [(b, 1) for b in high_priority] + [(b, 2) for b in medium_priority]
        location_groups = {}
        for bike, pr in all_bikes:
            loc = bike["location"]
            location_groups.setdefault(loc, []).append((bike, pr))

        routes = []
        unvisited = list(location_groups.keys())

        while unvisited:
            current_loc = self.maintenance_center
            route = []
            route_bikes = []

            while unvisited and len(route_bikes) < max_bikes_per_route:
                nearest_loc = min(unvisited, key=lambda loc: self.distance(current_loc, self.locations[loc]))
                for bike, pr in location_groups[nearest_loc]:
                    if len(route_bikes) < max_bikes_per_route:
                        route_bikes.append(bike)
                route.append(nearest_loc)
                current_loc = self.locations[nearest_loc]
                unvisited.remove(nearest_loc)

            route.append("Maintenance Center")
            routes.append({
                "route": route,
                "bikes": route_bikes,
                "estimated_time": self.calculate_route_time(route)
            })
        return routes

# =========================
# Alert System
# =========================
class MaintenanceAlertSystem:
    def __init__(self):
        self.alerts = deque(maxlen=200)
        self.last_alert_time = {}

    def _critical_alert(self, bike, comp_name, comp_risk):
        return {
            "type": "CRITICAL",
            "bike_id": bike["id"],
            "location": bike["location"],
            "component": comp_name,
            "risk_score": comp_risk,
            "temperature": bike["temperature"],
            "vibration": bike["vibration"],
            "pressure": bike["pressure"],
            "message": f"Bike {bike['id']} at {bike['location']} has {comp_name} failure risk of {comp_risk*100:.1f}%",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "read": False
        }

    def _warning_alert(self, bike, msg_component="Motor"):
        return {
            "type": "WARNING",
            "bike_id": bike["id"],
            "location": bike["location"],
            "component": msg_component,
            "risk_score": bike.get("overall_risk", 0.0),
            "temperature": bike["temperature"],
            "vibration": bike["vibration"],
            "pressure": bike["pressure"],
            "message": f"Bike {bike['id']} at {bike['location']}: high temperature ({bike['temperature']}°C)",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "read": False
        }

    def check_alerts(self, bikes):
        now = time.time()
        new_alerts = []
        for bike in bikes:
            bid = bike["id"]

            # Critical on high risk
            if bike.get("overall_risk", 0) > 0.8:
                if bid not in self.last_alert_time or (now - self.last_alert_time[bid] > 3600):
                    components = [
                        ("Brakes", bike["brakes_risk"]),
                        ("Tires", bike["tires_risk"]),
                        ("Chain", bike["chain_risk"]),
                        ("Gears", bike["gears_risk"]),
                        ("Electronics", bike["electronics_risk"]),
                    ]
                    comp, risk = max(components, key=lambda x: x[1])
                    alert = self._critical_alert(bike, comp, risk)
                    self.alerts.append(alert)
                    new_alerts.append(alert)
                    self.last_alert_time[bid] = now

            # Warning for high temperature
            elif bike["temperature"] > 85:
                if bid not in self.last_alert_time or (now - self.last_alert_time[bid] > 7200):
                    alert = self._warning_alert(bike)
                    self.alerts.append(alert)
                    new_alerts.append(alert)
                    self.last_alert_time[bid] = now
        return new_alerts

    def get_recent_alerts(self, limit=20):
        return list(self.alerts)[-limit:]

    def get_alerts_by_bike_id(self, bike_id):
        return [a for a in self.alerts if a["bike_id"] == bike_id]

    def mark_as_read(self, index):
        if 0 <= index < len(self.alerts):
            lst = list(self.alerts)
            lst[index]["read"] = True
            self.alerts.clear()
            self.alerts.extend(lst)

# =========================
# Session State Objects
# =========================
if "route_optimizer" not in st.session_state:
    st.session_state.route_optimizer = MaintenanceRouteOptimizer()

if "alert_system" not in st.session_state:
    st.session_state.alert_system = MaintenanceAlertSystem()

def generate_bike_data(n=20):
    bikes = []
    locations = list(st.session_state.route_optimizer.locations.keys())
    model_loaded = st.session_state.model_loaded
    scaler = st.session_state.scaler
    model = st.session_state.model

    for i in range(n):
        bike_id = f"BK{i+1:03d}"
        location = random.choice(locations)
        last_maint = datetime.now() - timedelta(days=random.randint(1, 180))
        temperature = random.randint(40, 100)
        vibration = random.randint(2, 15)
        pressure = random.randint(80, 120)
        anomaly_score = round(random.uniform(0.1, 0.9), 2)
        ride_duration = random.randint(30, 240)

        if model_loaded:
            X = np.array([[temperature, vibration, pressure, anomaly_score, ride_duration]])
            X_scaled = scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                probs = st.session_state.model(X_tensor).numpy()[0]
        else:
            if i % 5 == 0:
                probs = [random.uniform(0.8, 0.95) for _ in range(5)]
            else:
                probs = [random.random()*0.8 for _ in range(5)]

        bikes.append({
            "id": bike_id,
            "location": location,
            "last_maintenance": last_maint.strftime("%Y-%m-%d"),
            "days_since_maintenance": (datetime.now() - last_maint).days,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "anomaly_score": anomaly_score,
            "ride_duration": ride_duration,
            "brakes_risk": float(probs[0]),
            "tires_risk": float(probs[1]),
            "chain_risk": float(probs[2]),
            "gears_risk": float(probs[3]),
            "electronics_risk": float(probs[4]),
            "overall_risk": float(max(probs))
        })
    return bikes

if "bike_data" not in st.session_state:
    st.session_state.bike_data = generate_bike_data()

# Helper
def get_bike(bike_id: str):
    for b in st.session_state.bike_data:
        if b["id"].upper() == bike_id.upper():
            return b
    return None

# =========================
# Small UI Helpers
# =========================
def custom_progress_bar(value, label=""):
    progress_html = f"""
    <div>
        <div style="display: flex; justify-content: space-between;">
            <span>{label}</span>
            <span>{value*100:.1f}%</span>
        </div>
        <div class="custom-progress">
            <div class="custom-progress-bar"
                 style="width: {value*100}%;
                        background-color: {'#d62728' if value > 0.7 else '#ff7f0e' if value > 0.4 else '#2ca02c'};">
            </div>
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

def display_alert(alert, index=None):
    if alert["type"] == "CRITICAL":
        alert_class, alert_icon, alert_title = "alert-critical", "🔴", "CRITICAL"
    elif alert["type"] == "WARNING":
        alert_class, alert_icon, alert_title = "alert-warning", "🟠", "WARNING"
    else:
        alert_class, alert_icon, alert_title = "alert-info", "🔵", "INFO"

    button_key = f"read_{alert['bike_id']}_{alert['timestamp']}"
    alert_html = f"""
    <div class="{alert_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:1.2em;">{alert_icon}</span>
                <strong>{alert_title}: {alert['bike_id']} at {alert['location']}</strong>
            </div>
            <span style="font-size:0.85em;color:#777;">{alert['timestamp']}</span>
        </div>
        <div style="display:grid;grid-template-columns:auto auto auto;gap:8px 15px;font-size:0.95em;margin-bottom:10px;">
            <div><strong>Component:</strong> {alert['component']}</div>
            <div><strong>Risk:</strong> {alert['risk_score']*100:.1f}%</div>
            <div><strong>Temp:</strong> {alert['temperature']}°C</div>
        </div>
        <div style="display:flex;gap:8px;">
            <button style="padding:4px 8px;background:#d32f2f;color:#fff;border:none;border-radius:4px;font-size:0.85em;cursor:pointer;"
                onclick="alert('Maintenance scheduled for {alert['bike_id']}')">Schedule</button>
            <button style="padding:4px 8px;background:#f5f5f5;color:#333;border:1px solid #ddd;border-radius:4px;font-size:0.85em;cursor:pointer;"
                onclick="alert('Detailed report generated for {alert['bike_id']}')">Details</button>
        </div>
    </div>
    """
    st.markdown(alert_html, unsafe_allow_html=True)

    if index is not None and not alert["read"]:
        if st.button("Mark as Read", key=button_key, type="secondary"):
            st.session_state.alert_system.mark_as_read(index)
            st.rerun()

# =========================
# Header
# =========================
st.markdown('<h1 class="main-header">🚲 e-Bike Maintenance Intelligence Dashboard</h1>', unsafe_allow_html=True)

# =========================
# Search Box (Bike ID → Alerts)
# =========================
st.markdown('<div class="search-box">', unsafe_allow_html=True)
st.subheader("🔍 Search for Specific Bike")
scol1, scol2 = st.columns([2, 1])
with scol1:
    bike_id_search = st.text_input("Enter Bike ID (e.g., BK001, BK002):", placeholder="BK001")
with scol2:
    search_button = st.button("Search Alerts", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Sidebar: Single Bike Assessment (Manual or by Bike ID)
# =========================
with st.sidebar:
    st.header("🔧 Single Bike Assessment")
    with st.expander("📊 Feature Importance", expanded=False):
        st.markdown("""
        - **Temperature (°C):** Motor overheating → sudden failures.
        - **Vibration (mm/s):** Loose/imbalanced parts.
        - **Tire Pressure (psi):** Handling + wear.
        - **Anomaly Score:** Unusual sensor patterns.
        - **Ride Duration (mins):** Cumulative stress.
        """)

    if not st.session_state.model_loaded:
        st.warning("Model not found — running in demo mode with synthetic predictions.")

    mode = st.radio("Input Mode", ["Manual Inputs", "Use Existing Bike ID"], index=1)
    selected_bike_id = None
    bike_prefill = None

    if mode == "Use Existing Bike ID":
        ids = [b["id"] for b in st.session_state.bike_data]
        selected_bike_id = st.selectbox("Select Bike ID", ids)
        bike_prefill = get_bike(selected_bike_id)

    def slider_val(name, mn, mx, default):
        return st.slider(name, mn, mx, default)

    if bike_prefill:
        temperature = slider_val("Temperature (°C)", 0, 120, int(bike_prefill["temperature"]))
        vibration  = slider_val("Vibration (mm/s)", 0, 20, int(bike_prefill["vibration"]))
        pressure   = slider_val("Tire Pressure (psi)", 50, 200, int(bike_prefill["pressure"]))
        anomaly_score = st.slider("Anomaly Score", 0.0, 1.0, float(bike_prefill["anomaly_score"]))
        ride_duration = slider_val("Ride Duration (mins)", 0, 300, int(bike_prefill["ride_duration"]))
    else:
        temperature = slider_val("Temperature (°C)", 0, 120, 70)
        vibration  = slider_val("Vibration (mm/s)", 0, 20, 5)
        pressure   = slider_val("Tire Pressure (psi)", 50, 200, 100)
        anomaly_score = st.slider("Anomaly Score", 0.0, 1.0, 0.5)
        ride_duration = slider_val("Ride Duration (mins)", 0, 300, 60)

    predict_clicked = st.button("🔮 Predict Failure Risks", use_container_width=True)

    if predict_clicked:
        # Prepare input
        X = np.array([[temperature, vibration, pressure, anomaly_score, ride_duration]])
        if st.session_state.model_loaded:
            X_scaled = st.session_state.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                probs = st.session_state.model(X_tensor).numpy()[0]
        else:
            probs = np.array([random.random() for _ in range(5)])
            if probs.max() > 0:
                probs = probs / probs.max()  # normalize to [0,1]

        parts = ["Brakes", "Tires", "Chain", "Gears", "Electronics"]
        colors = ["#ff6b6b", "#ffa726", "#66bb6a", "#42a5f5", "#ab47bc"]

        st.subheader("Prediction Results")
        fig = go.Figure(go.Bar(
            x=probs.tolist(), y=parts, orientation='h',
            marker_color=colors, text=[f"{p*100:.1f}%" for p in probs], textposition='auto'
        ))
        fig.update_layout(title="Component Failure Risks", xaxis_title="Failure Probability",
                          xaxis=dict(range=[0, 1]), height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detailed Risk Assessment")
        for p_name, pr in zip(parts, probs):
            cls = "risk-high" if pr > 0.7 else "risk-medium" if pr > 0.4 else "risk-low"
            st.markdown(f'<div class="component-card">{p_name}: <span class="{cls}">{pr*100:.1f}%</span></div>',
                        unsafe_allow_html=True)
            custom_progress_bar(pr, p_name)

        # If a Bike ID was chosen, update that bike & generate an alert for it
        if bike_prefill:
            bike_prefill["temperature"] = int(temperature)
            bike_prefill["vibration"] = int(vibration)
            bike_prefill["pressure"] = int(pressure)
            bike_prefill["anomaly_score"] = float(anomaly_score)
            bike_prefill["ride_duration"] = int(ride_duration)
            bike_prefill["brakes_risk"] = float(probs[0])
            bike_prefill["tires_risk"] = float(probs[1])
            bike_prefill["chain_risk"] = float(probs[2])
            bike_prefill["gears_risk"] = float(probs[3])
            bike_prefill["electronics_risk"] = float(probs[4])
            bike_prefill["overall_risk"] = float(probs.max())

            # Force an alert check for this specific bike
            generated = st.session_state.alert_system.check_alerts([bike_prefill])
            if generated:
                st.success(f"Alert(s) generated for {bike_prefill['id']}. See 'Active Alerts' or search above.")
            else:
                st.info(f"No alert threshold reached for {bike_prefill['id']}.")

# =========================
# Main: Fleet Overview
# =========================
st.header("🏍️ Fleet Overview")
# Alert pass on the full dataset each run (simulate streaming checks)
new_alerts = st.session_state.alert_system.check_alerts(st.session_state.bike_data)

# Search results first (Bike ID)
if search_button and bike_id_search:
    alerts = st.session_state.alert_system.get_alerts_by_bike_id(bike_id_search.upper())
    if alerts:
        st.header(f"🚨 Alerts for {bike_id_search.upper()}")
        critical = [a for a in alerts if a["type"] == "CRITICAL"]
        warning = [a for a in alerts if a["type"] == "WARNING"]

        if critical:
            st.subheader(f"Critical Issues ({len(critical)})")
            for a in critical:
                display_alert(a)
        if warning:
            st.subheader(f"Warnings ({len(warning)})")
            for a in warning:
                display_alert(a)
    else:
        st.info(f"No alerts found for bike {bike_id_search.upper()}")
elif new_alerts:
    st.header("🚨 Active Alerts")
    critical = [a for a in new_alerts if a["type"] == "CRITICAL"]
    warning = [a for a in new_alerts if a["type"] == "WARNING"]
    if critical:
        st.subheader(f"Critical Issues ({len(critical)})")
        for a in critical:
            display_alert(a)
    if warning:
        st.subheader(f"Warnings ({len(warning)})")
        for a in warning:
            display_alert(a)

# =========================
# Key Metrics
# =========================
data = st.session_state.bike_data
high_risk_bikes = sum(1 for b in data if b["overall_risk"] > 0.7)
medium_risk_bikes = sum(1 for b in data if 0.4 <= b["overall_risk"] <= 0.7)
avg_temp = sum(b['temperature'] for b in data) / len(data)
avg_risk = sum(b['overall_risk'] for b in data) / len(data)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Bikes", len(data))
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("High Risk Bikes", high_risk_bikes, delta=f"{high_risk_bikes/len(data)*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Avg. Temperature", f"{avg_temp:.1f}°C")
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Avg. Risk", f"{avg_risk*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# Risk distribution
risk_values = [b["overall_risk"] for b in data]
fig_hist = px.histogram(x=risk_values, nbins=10, title="Risk Distribution Across Fleet",
                        labels={"x": "Risk Score", "y": "Number of Bikes"})
fig_hist.update_layout(bargap=0.1)
st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Bike Status", "📊 Sensor Analytics", "🗺️ Location Overview", "🛣️ Maintenance Routes", "📋 Alert History"])

with tab1:
    # Show a pretty table; keep numerics for plots elsewhere
    df_num = pd.DataFrame(data)
    df_view = df_num.copy()
    risk_cols = ["brakes_risk", "tires_risk", "chain_risk", "gears_risk", "electronics_risk", "overall_risk"]
    for c in risk_cols:
        df_view[c] = df_view[c].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(df_view, use_container_width=True)

with tab2:
    df_num = pd.DataFrame(data)
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(df_num, x="temperature", y="overall_risk",
                          title="Temperature vs Risk Score",
                          labels={"temperature": "Temperature (°C)", "overall_risk": "Risk Score"},
                          hover_data=["id", "location"])
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.scatter(df_num, x="vibration", y="overall_risk",
                          title="Vibration vs Risk Score",
                          labels={"vibration": "Vibration (mm/s)", "overall_risk": "Risk Score"},
                          hover_data=["id", "location"])
        st.plotly_chart(fig2, use_container_width=True)

    corr_matrix = df_num[["temperature", "vibration", "pressure", "anomaly_score", "ride_duration", "overall_risk"]].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    location_counts = {}
    location_risks = {}
    for b in data:
        loc = b["location"]
        location_counts[loc] = location_counts.get(loc, 0) + 1
        location_risks.setdefault(loc, []).append(b["overall_risk"])
    avg_risk_per_location = {loc: sum(v)/len(v) for loc, v in location_risks.items()}

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(x=list(location_counts.keys()), y=list(location_counts.values()),
                     title="Bike Distribution by Location",
                     labels={"x": "Location", "y": "Number of Bikes"})
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(x=list(avg_risk_per_location.keys()), y=list(avg_risk_per_location.values()),
                     title="Average Risk by Location",
                     labels={"x": "Location", "y": "Average Risk Score"})
        fig.update_layout(xaxis_tickangle=45, yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("🛣️ Optimized Maintenance Routes")
    high_priority = [b for b in data if b["overall_risk"] > 0.7]
    medium_priority = [b for b in data if 0.4 <= b["overall_risk"] <= 0.7]

    if high_priority or medium_priority:
        routes = st.session_state.route_optimizer.optimize_routes(high_priority, medium_priority)
        st.subheader(f"Generated {len(routes)} Maintenance Routes")

        for i, r in enumerate(routes):
            with st.expander(f"Route {i+1} - Estimated Time: {r['estimated_time']} minutes"):
                st.write("**Route Path:**")
                st.write(" → ".join(r["route"]))
                st.write("**Bikes to Service:**")
                for b in r["bikes"]:
                    cls = "risk-high" if b["overall_risk"] > 0.7 else "risk-medium"
                    st.markdown(
                        f"- {b['id']} at {b['location']} "
                        f"(<span class='{cls}'>{b['overall_risk']*100:.1f}% risk</span>)",
                        unsafe_allow_html=True
                    )

        st.subheader("Route Visualization")
        ro = st.session_state.route_optimizer
        fig = go.Figure()

        # Maintenance center
        fig.add_trace(go.Scatter(
            x=[ro.maintenance_center[0]], y=[ro.maintenance_center[1]],
            mode='markers', marker=dict(size=15, color='red'), name='Maintenance Center'
        ))
        # Locations
        for loc, coords in ro.locations.items():
            fig.add_trace(go.Scatter(
                x=[coords[0]], y=[coords[1]], mode='markers+text',
                marker=dict(size=10, color='blue'),
                text=loc, textposition="top center", name=loc
            ))
        # Draw each route
        for i, r in enumerate(routes):
            coords = [ro.maintenance_center]
            for loc in r["route"]:
                if loc != "Maintenance Center":
                    coords.append(ro.locations[loc])
            coords.append(ro.maintenance_center)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                                     line=dict(width=2, dash='dash'),
                                     name=f'Route {i+1}'))
        fig.update_layout(title="Maintenance Routes Visualization",
                          xaxis_title="X Coordinate", yaxis_title="Y Coordinate",
                          showlegend=True, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bikes currently require maintenance. All systems are operational!")

with tab5:
    st.header("📋 Alert History")
    recent = st.session_state.alert_system.get_recent_alerts(50)
    if recent:
        c1, c2, c3 = st.columns(3)
        with c1:
            filter_type = st.selectbox("Filter by type", ["All", "CRITICAL", "WARNING"])
        with c2:
            bike_filter = st.text_input("Filter by Bike ID", placeholder="e.g., BK001")
        with c3:
            show_read = st.checkbox("Show read alerts", value=False)

        filtered = recent
        if filter_type != "All":
            filtered = [a for a in filtered if a["type"] == filter_type]
        if bike_filter:
            filtered = [a for a in filtered if bike_filter.upper() in a["bike_id"]]
        if not show_read:
            filtered = [a for a in filtered if not a["read"]]

        if filtered:
            st.write(f"Showing {len(filtered)} alerts")
            # Show newest first
            for i, alert in enumerate(reversed(filtered)):
                original_index = len(recent) - 1 - i
                display_alert(alert, original_index)
        else:
            st.info("No alerts match your filters.")
    else:
        st.info("No alerts in the recent history. All systems are operational!")

# =========================
# Maintenance Recommendations
# =========================
st.header("🛠️ Maintenance Recommendations")
high_list = [b for b in data if b["overall_risk"] > 0.7]
med_list = [b for b in data if 0.4 <= b["overall_risk"] <= 0.7]

if high_list:
    st.error("Immediate attention required for the following bikes:")
    for bike in high_list:
        comp_risks = [
            ("Brakes", bike["brakes_risk"]),
            ("Tires", bike["tires_risk"]),
            ("Chain", bike["chain_risk"]),
            ("Gears", bike["gears_risk"]),
            ("Electronics", bike["electronics_risk"]),
        ]
        top_comp, top_score = max(comp_risks, key=lambda x: x[1])
        st.write(f"- **{bike['id']}** at {bike['location']} — {top_comp} risk: **{top_score*100:.1f}%**")

if med_list:
    st.warning("Schedule preventive maintenance for these bikes soon:")
    for bike in med_list:
        st.write(f"- **{bike['id']}** at {bike['location']} — Overall risk: **{bike['overall_risk']*100:.1f}%**")

if not high_list and not med_list:
    st.success("All bikes are in good condition. No immediate maintenance required.")
