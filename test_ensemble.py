import sys
sys.path.append("src")

from data_loader import load_city_data
from models.ensemble_forecast import ensemble_forecast

city = "ahmedabad"

print("🔍 Loading city data...")
df = load_city_data(city)

print("\n📊 Running ensemble forecast...")
out = ensemble_forecast(df, city)

print("\n✅ Here is out.head():")
print(out.head())
