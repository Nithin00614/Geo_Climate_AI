import numpy as np
import pandas as pd

def simple_forecast(series, days=14):
    """Generate forecast with trend and seasonal variation."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    
    # Calculate trend
    if len(s) >= 30:
        slope = (s.iloc[-1] - s.iloc[-30]) / 30
    elif len(s) >= 2:
        slope = (s.iloc[-1] - s.iloc[0]) / max(len(s) - 1, 1)
    else:
        slope = 0
    
    last_value = float(s.iloc[-1])
    
    # Calculate seasonal component (weekly pattern)
    if len(s) >= 7:
        weekly = s.iloc[-7:].values
    else:
        weekly = np.repeat(last_value, 7)
    
    # Calculate residual std for variation
    if len(s) > 1:
        residual_std = float(np.std(np.diff(s.values)))
    else:
        residual_std = 0.5
    
    forecast = []
    ci_lower = []
    ci_upper = []
    
    for i in range(1, days + 1):
        # Trend component
        trend_val = last_value + slope * i
        
        # Seasonal component (weekly pattern)
        seasonal_idx = (i - 1) % 7
        seasonal_val = weekly[seasonal_idx] - np.mean(weekly)
        
        # Random walk component
        noise = np.random.normal(0, residual_std * 0.2)
        
        # Combine
        pred = trend_val + seasonal_val * 0.3 + noise
        forecast.append(pred)
        
        # Confidence intervals (wider for longer forecasts)
        margin = 1.2 + (i / days) * 0.8
        ci_lower.append(pred - margin)
        ci_upper.append(pred + margin)
    
    return pd.DataFrame({
        "forecast": forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    })
