# e-Bike Predictive Maintenance Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.png)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red.png)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.png)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.png)](LICENSE)

## Project Overview

This project implements a comprehensive predictive maintenance system for a public e-bike sharing fleet. It uses machine learning to predict component-specific failures (brakes, tires, chain, gears, electronics) based on sensor data, enabling preventive maintenance to improve bike availability and safety.

Key features include:
- **Real-time Risk Prediction**: Multi-label classification model (MLP neural network) to estimate failure probabilities for bike components.
- **Interactive Dashboard**: Built with Streamlit, featuring fleet overview, single-bike assessment, alerts, route optimization, visualizations, and maintenance recommendations.
- **Alert System**: Generates critical/warning alerts for high-risk bikes or anomalies (e.g., high temperature).
- **Route Optimization**: Simple distance-based optimizer for maintenance crew routes, prioritizing high-risk bikes.
- **Data-Driven Insights**: Visualizations for risk distribution, sensor correlations, and location-based analytics.

The system aligns with the problem statement's key modules: telemetry ingestion (simulated via data), feature engineering, multi-label model, operations dashboard, historical data management, and external integrations (simulated weather/topography).

## Versions

The dashboard evolved through iterative improvements:
- **app.py**: Basic single-bike predictor with sliders and bar chart.
- **app2.py**: Added fleet overview table, metrics, and explanations for high-risk bikes.
- **app3.py**: Enhanced with Plotly visualizations, tabs for organization, and improved UI styling.
- **app4.py**: Professional version with advanced features like searchable alerts, route visualization, sidebar assessment modes, and session-state managed objects for alerts/optimization.

Use `app4.py` for the latest, production-ready version.

## Data Sources

Data is stored in the `data/` folder:
- `api_data_generated.csv`: Simulated real-time sensor telemetry (temperature, vibration, pressure, status, anomaly_score, predicted_failure_in_days).
- `maintenance_log_enhanced.csv`: Historical maintenance records with component replacements and downtime.
- `historical_maintenance_1yr_generated.csv`: One-year historical data for training/validation.

These CSVs were used to train the model. The data is synthetic but mimics real e-bike sensor patterns.

## Model Details

- **Architecture**: Multi-Layer Perceptron (MLP) with 2 hidden layers (32 and 16 neurons), ReLU activation, and dropout (0.3) for regularization.
- **Input Features**: Temperature (°C), Vibration (mm/s), Tire Pressure (psi), Anomaly Score (0-1), Ride Duration (mins).
- **Outputs**: Sigmoid probabilities for 5 components (brakes, tires, chain, gears, electronics).
- **Training**: Binary Cross-Entropy loss, Adam optimizer. Trained on aggregated sensor data linked to maintenance labels.
- **Files**:
  - `bike_model.pth`: Trained model weights.
  - `scaler.pkl`: StandardScaler for feature normalization.
- **Training Script**: `train_model.py` - Processes data, trains the model, and saves artifacts.

## Requirements

- Python 3.8+
- Libraries: Install via `pip install -r requirements.txt` (create one if not present):
  ```
  streamlit==1.38.0
  torch==2.4.0
  numpy==1.24.3
  pandas==2.0.3
  joblib==1.4.2
  matplotlib==3.7.2
  plotly==5.23.0
  ```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bike-maintenance-dashboard.git
   cd bike-maintenance-dashboard
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place data files in `data/` folder (or update paths in code).

4. (Optional) Train the model:
   ```
   python train_model.py
   ```
   This generates `bike_model.pth` and `scaler.pkl`.

## Usage

### Running the Dashboard
Run the latest version:
```
streamlit run app4.py
```

- **Access**: Open in browser (default: http://localhost:8501).
- **Features**:
  - **Fleet Overview**: Metrics, tables, and visualizations.
  - **Single Bike Predictor**: Manual inputs or select existing bike ID for predictions.
  - **Alerts**: Search by ID or view recent; mark as read.
  - **Routes**: Optimized paths for high/medium-risk bikes.
  - **History**: Filtered alert logs.

### Training the Model
- Run `train_model.py` to process data and train/save the model.
- Customize hyperparameters in the script (e.g., epochs, batch size).

### Demo Mode
If model files are missing, the dashboard falls back to synthetic/random predictions for demonstration.

## Project Structure

```
bike_share/
├── data/
│   ├── api_data_generated.csv
│   ├── maintenance_log_enhanced.csv
│   └── historical_maintenance_1yr_generated.csv
├── app.py          # Version 1: Basic predictor
├── app2.py         # Version 2: Added fleet overview
├── app3.py         # Version 3: Enhanced visualizations
├── app4.py         # Version 4: Professional dashboard
├── train_model.py  # Model training script
├── bike_model.pth  # Trained model weights
├── scaler.pkl      # Feature scaler
├── requirements.txt # Dependencies
└── README.md       # This file
```

## Limitations & Improvements
- **Data**: Synthetic; integrate real telemetry for production.
- **Model**: Basic MLP; consider LSTM for time-series or ensemble methods.
- **Optimization**: Simple greedy route optimizer; integrate OR-Tools for TSP solving.
- **Real-time**: Simulate streaming; add Kafka/WebSockets for live data.
- **Security**: Add authentication for production deployment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/), and [Plotly](https://plotly.com/).
- Inspired by predictive maintenance use cases in IoT.

For questions or contributions, open an issue or PR!
