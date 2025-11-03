---

 🌍 GeoClimate-AI

AI-Powered Climate Risk & Weather Prediction System

GeoClimate-AI is an intelligent weather and climate analysis project that collects real-time data, preprocesses it, and trains an AI model to predict temperature trends for different cities.
It’s the foundation for a future Geo-AI system capable of forecasting extreme climate risks such as floods, droughts, and heatwaves.

---

 🚀 Features

  🌦️ Fetches live weather data from the OpenWeather API
  🧹 Preprocesses and cleans the data automatically
  🧠 Trains AI model (currently Linear Regression, soon upgraded to Random Forest & LSTM)
  💾 Saves trained model (`temperature_model.pkl`) for future predictions
  🔍 Predicts temperature using humidity, pressure, and wind speed
  📈 Will include visualizations and a Streamlit dashboard** in the next phase

---

 🧰 Tech Stack

| Category                   | Tools & Libraries           |
| -------------------------- | --------------------------- |
| Language                   | Python                      |
| AI / ML                    | scikit-learn, pandas, numpy |
| Data Fetching              | OpenWeather API, requests   |
| Visualization (Next Phase) | matplotlib, seaborn, folium |
| Web App (Next Phase)       | Streamlit                   |
| Environment                | VS Code + virtualenv        |
| Version Control            | Git + GitHub                |

---

 📂 Project Structure

```
geoclimate-ai/
│
├── data/                      # Weather data CSV files
├── models/                    # Saved model (.pkl)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Fetches data from OpenWeather API
│   ├── preprocess.py          # Cleans and prepares dataset
│   ├── model.py               # Trains ML model & saves it
│   ├── predict.py             # Predicts temperature using saved model
│
├── .env                       # Contains your API key
├── .gitignore
├── README.md
└── main.py                    # Entry point of the project
```

---

⚙️ Setup & Installation

1. Clone the repository

   ```bash
   git clone https://github.com/<your-username>/geoclimate-ai.git
   cd geoclimate-ai
   ```

2. Create a virtual environment

   ```bash
   python -m venv venv
   venv\Scripts\activate  # (Windows)
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

   *(If you don’t have one yet, we can create it tomorrow.)*

4. Add your OpenWeather API key
   Create a file named `.env` and add:

   ```
   OPENWEATHER_API_KEY=your_api_key_here
   ```

5. Run the project

   ```bash
   python main.py
   ```

---

 🧪 Current Progress

✅ Weather data fetching from OpenWeather API
✅ Preprocessing pipeline built
✅ Linear Regression model trained & evaluated
✅ Model successfully saved & used for prediction
🔄 Next: Visualization + Streamlit dashboard

---

 📊 Example Output

```
🤖 Training temperature prediction model...
✅ Model trained successfully!
📊 Mean Absolute Error: 1.52
📈 R² Score: -0.24
💾 Model saved successfully!

🌡️ Testing saved model for prediction...
🤖 Predicted Temperature: 26.31 °C
🎯 All steps completed successfully!
```

---

🌱 Next Development Phases

1. Upgrade ML model → RandomForest / LSTM for time-series forecasting
2. Visualization → Real-time trend plots using matplotlib
3. Streamlit App → Interactive city-based prediction dashboard
4. Geo-Analytics → Integrate geospatial features using GeoPandas & Folium

---
👨‍💻 Author

Nithin Gowda

