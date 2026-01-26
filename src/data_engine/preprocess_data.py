import pandas as pd
import os

DATA_PATH = "data/real_weather/"
OUTPUT_PATH = "data/processed/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def preprocess_all():
    for file in os.listdir(DATA_PATH):
        if not file.endswith(".csv"):
            continue
        
        df = pd.read_csv(os.path.join(DATA_PATH, file))
        
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        
        df = df.sort_values("date")
        df = df.rename(columns={"temperature": "temp"})
        
        city = file.replace(".csv", "").title()
        df.to_csv(os.path.join(OUTPUT_PATH, f"{city}.csv"), index=False)
        print(f"✔ Processed: {city}")

if __name__ == "__main__":
    preprocess_all()
