# GeoClimate AI

> AI-powered weather forecasting and climate risk analytics platform using ensemble machine learning.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.39+-green.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **🔮 14-Day Ensemble Forecasts** - LSTM + Prophet + XGBoost ensemble predictions
- **📡 IoT Sensor Monitoring** - Real-time sensor data collection and storage
- **📊 Climate Risk Analytics** - Multi-factor risk scoring and severity classification
- **🤖 XAI Explainability** - SHAP-based model interpretability
- **📈 Model Management** - Train, evaluate, and manage per-city models
- **📧 Alert System** - Email notifications for high-risk events
- **📅 Daily Reports** - Export forecast summaries in CSV/HTML
- **🌍 38+ Cities** - Pre-processed datasets for major global cities

## Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/geoclimate-ai.git
cd geoclimate-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run src/app.py
```

The app opens at `http://localhost:8501`

## Usage

### Dashboard Pages

| Page | Purpose |
|------|---------|
| **Home** | System overview and statistics |
| **Weather Forecast** | 14-day temperature predictions |
| **IoT Dashboard** | Real-time sensor data monitoring |
| **Risk Analytics** | Climate risk scoring and alerts |
| **Ensemble Insights** | Model comparison and weights |
| **XAI Explainability** | SHAP feature importance |
| **Model Accuracy** | Performance backtesting |
| **Model Manager** | Train and manage models |
| **Auto Retrain** | Scheduled model updates |
| **Database Viewer** | Query sensor data |
| **Daily Reports** | Generate forecast summaries |

## Architecture

### Machine Learning Models

**Ensemble Strategy**
- LSTM Neural Network (7-day sequences → 14-day forecast)
- XGBoost (Lag-based gradient boosting)
- Facebook Prophet (Time series with seasonality)
- **Weighting:** Inverse variance of model predictions

**Fallback Methods**
- Simple forecast (trend + seasonal + stochastic)
- Moving average
- Linear trend
- Persistence

### Data Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Models → Ensemble → Forecast
                ↓                                              ↓
           SQLite DB                                      Risk Analysis
```

## Project Structure

```
geoclimate-ai/
├── src/
│   ├── app.py                    # Main Streamlit application
│   ├── iot_simulator.py          # IoT data simulator
│   ├── models/                   # ML models
│   │   ├── lstm_model.py
│   │   ├── xgb_model.py
│   │   ├── prophet_model.py
│   │   ├── forecast_engine.py    # Ensemble orchestrator
│   │   └── *.h5, *.joblib       # Trained models & scalers
│   ├── pages/                    # 11 Streamlit pages
│   ├── data_engine/              # Data processing
│   │   ├── climate_risk_full.py
│   │   ├── preprocess_data.py
│   │   └── email_engine.py
│   └── utils/                    # Shared utilities
├── data/
│   ├── processed/                # 38 city datasets
│   ├── plots/                    # Generated plots
│   └── alerts_log.txt
├── requirements.txt
└── config.toml
```

## Data Format

### Input CSV
```csv
date,temperature,humidity,pressure,wind_speed,aqi
2024-01-01,25.5,65.0,1013.25,5.2,45
```

**Required:** `date`, `temperature`  
**Optional:** `humidity`, `pressure`, `wind_speed`, `aqi`

### Output Forecast
```python
{
  'date': DatetimeIndex,
  'final_pred': ndarray,         # Ensemble forecast
  'lstm_weight': float,           # LSTM weight
  'prophet_weight': float,        # Prophet weight
  'xgb_weight': float             # XGBoost weight
}
```

## Configuration

Edit `config.toml` to customize:
```toml
[cities]
cities = ["Bengaluru", "Delhi", "Mumbai"]

[model]
lstm_epochs = 30
xgb_rounds = 100
forecast_horizon = 14

[risk]
temp_threshold = 35
humidity_threshold = 80
aqi_threshold = 300
```

## Tech Stack

- **Framework:** Streamlit 1.39
- **Deep Learning:** TensorFlow/Keras
- **Gradient Boosting:** XGBoost
- **Time Series:** Facebook Prophet
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Visualization:** Plotly, Matplotlib
- **Explainability:** SHAP
- **Database:** SQLite
- **Visualization Library:** Seaborn

## Installation for Development

```bash
# Install with training dependencies
pip install -r requirements_training.txt

# Run tests
python test_predictions.py

# Generate mock data
python generate_mock_data.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No data available | Place CSV files in `data/processed/` or run `python generate_mock_data.py` |
| TensorFlow warnings | Set `TF_ENABLE_ONEDNN_OPTS=0` environment variable |
| Model files not found | Train models in Model Manager page |
| Import errors | Run from project root with `src/` folder present |

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{geoclimate_ai,
  title={GeoClimate AI: Weather Forecasting with Ensemble ML},
  author={Gowda, Nithin},
  year={2026},
  url={https://github.com/yourusername/geoclimate-ai}
}
```

## Acknowledgments

- TensorFlow/Keras team for deep learning framework
- Facebook Prophet for time series forecasting
- XGBoost team for gradient boosting
- Streamlit team for excellent web framework
- SHAP team for model explainability

---

**Status:** Production Ready ✅  
**Version:** 1.0.0  
**Last Updated:** January 2026

