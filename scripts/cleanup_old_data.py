import os

folder = "data"
for f in os.listdir(folder):
    if f.endswith(".csv") and f != "real_weather":
        print("Deleting:", f)
        os.remove(os.path.join(folder, f))

print("✅ Old synthetic CSV files deleted safely.")
