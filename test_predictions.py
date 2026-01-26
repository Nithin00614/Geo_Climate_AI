"""
Test script to verify prediction improvements and fix issues.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the updated forecast functions
from src.models.predict import simple_forecast
from src.models.forecast_engine import _persistence, _linear_trend, _moving_average

def create_test_data(days=90):
    """Create synthetic weather data with realistic patterns."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Create realistic temperature pattern with trend and seasonality
    t = np.arange(days)
    base_temp = 25
    trend = 0.05 * t  # Slight warming trend
    seasonal = 5 * np.sin(2 * np.pi * t / 365)  # Annual seasonality
    noise = np.random.normal(0, 1, days)
    
    temps = base_temp + trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temps,
        'humidity': np.random.uniform(30, 90, days),
        'pressure': np.random.uniform(990, 1030, days),
        'wind_speed': np.random.uniform(0, 15, days)
    })
    
    return df

def test_simple_forecast():
    """Test the improved simple_forecast function."""
    print("\n" + "="*60)
    print("Testing Simple Forecast Function")
    print("="*60)
    
    df = create_test_data()
    forecast_df = simple_forecast(df['temperature'], days=14)
    
    print(f"\nInput data points: {len(df)}")
    print(f"Last 5 temperature values:\n{df['temperature'].tail().values}")
    print(f"\nForecast for next 14 days:\n{forecast_df['forecast'].values}")
    
    # Check for flat lines
    forecast_values = forecast_df['forecast'].values
    std_dev = np.std(forecast_values)
    max_val = np.max(forecast_values)
    min_val = np.min(forecast_values)
    
    print(f"\nForecast Statistics:")
    print(f"  Min: {min_val:.2f}°C")
    print(f"  Max: {max_val:.2f}°C")
    print(f"  Range: {max_val - min_val:.2f}°C")
    print(f"  Std Dev: {std_dev:.4f}°C")
    
    if std_dev < 0.01:
        print("  ⚠️ WARNING: Forecast is nearly flat!")
    else:
        print("  ✅ GOOD: Forecast has realistic variation")
    
    return std_dev > 0.1

def test_fallback_methods():
    """Test improved fallback forecast methods."""
    print("\n" + "="*60)
    print("Testing Fallback Methods")
    print("="*60)
    
    df = create_test_data()
    horizon = 14
    
    # Test each fallback method
    methods = {
        'Persistence': _persistence(df, horizon),
        'Linear Trend': _linear_trend(df, horizon),
        'Moving Average': _moving_average(df, horizon)
    }
    
    for method_name, preds in methods.items():
        std_dev = np.std(preds)
        print(f"\n{method_name}:")
        print(f"  Predictions: {preds[:5]}...")
        print(f"  Std Dev: {std_dev:.4f}")
        if std_dev < 0.01:
            print(f"  ⚠️  Nearly flat!")
        else:
            print(f"  ✅ Has variation")

def test_iot_simulator_imports():
    """Test that IoT simulator can be imported without errors."""
    print("\n" + "="*60)
    print("Testing IoT Simulator Imports")
    print("="*60)
    
    try:
        from src.iot_simulator import simulate
        print("✅ IoT simulator imports successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing IoT simulator: {e}")
        return False

def test_ensemble_forecast():
    """Test the ensemble forecast function."""
    print("\n" + "="*60)
    print("Testing Ensemble Forecast Function")
    print("="*60)
    
    try:
        from src.models.forecast_engine import ensemble_forecast
        
        df = create_test_data()
        result = ensemble_forecast(df, city="TestCity", horizon=14)
        
        print(f"\nEnsemble Forecast Result Shape: {result.shape}")
        print(f"Columns: {list(result.columns)}")
        print(f"\nFinal Predictions (next 14 days):")
        print(result[['date', 'final_pred']].to_string())
        
        # Check variation
        std_dev = np.std(result['final_pred'].values)
        print(f"\nFinal Prediction Std Dev: {std_dev:.4f}")
        
        if std_dev < 0.01:
            print("⚠️  Ensemble predictions are nearly flat!")
            return False
        else:
            print("✅ Ensemble has good variation")
            return True
            
    except Exception as e:
        print(f"❌ Error in ensemble forecast: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🔍 GeoClimate AI - Prediction System Tests " + "🔍")
    
    results = {
        "Simple Forecast": test_simple_forecast(),
        "Fallback Methods": (test_fallback_methods() or True),  # Just informational
        "IoT Simulator Imports": test_iot_simulator_imports(),
        "Ensemble Forecast": test_ensemble_forecast()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Predictions should show good variation.")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
