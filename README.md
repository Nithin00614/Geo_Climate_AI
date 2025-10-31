🛰️ GeoClimate-AI

An intelligent system to collect, store, and analyze global weather data using the OpenWeather API.

 🌦️ Overview

GeoClimate-AI is a data-driven weather intelligence tool that fetches live weather information from the 'OpenWeather API', stores it locally, and prepares it for machine learning analysis.
It’s designed as the foundation for future climate trend prediction models.

⚙️ Features

✅ Fetches real-time weather data (temperature, humidity, pressure, wind speed, etc.)
✅ Automatically stores weather logs into `data/weather_data.csv`
✅ Modular structure — easy to extend for ML training or forecasting
✅ Uses `.env` for secure API key management

🧩 Project Structure

GeoClimate-AI/
│
├── src/
│   ├── data_loader.py      # Fetches and saves weather data
│   ├── preprocess.py       # (Future) Data cleaning & transformation
│   ├── model.py            # (Future) ML model training & evaluation
│   └── __init__.py
│
├── data/
│   └── weather_data.csv    # Auto-generated weather logs
│
├── .env                    # Contains your OpenWeather API key
├── main.py                 # Entry point for running the app
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md


 🚀 Setup Instructions

1. Clone the repository

   bash
   git clone https://github.com/Nithin00614/GeoClimate-AI.git
   cd GeoClimate-AI
   

2. Create a virtual environment

   bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate   # On macOS/Linux
   

3. Install dependencies

   bash
   pip install -r requirements.txt
   

4. Set up `.env` file

   
   OPENWEATHER_API_KEY=your_api_key_here
   

5. Run the project

   bash
   python main.py
   



🧠 Tech Stack

* Python 3.10+
* Requests for API calls
* Pandas for data handling
* dotenv for environment management
* (Planned) Scikit-learn / TensorFlow for climate predictions


 🧭 Next Steps

* Add preprocessing pipeline (`src/preprocess.py`)
* Build ML model to predict temperature trends
* Deploy via Streamlit or Flask dashboard
* Automate data collection using schedulers or cron jobs


 👨‍💻 Author

Nithin Gowda
📫 [Your LinkedIn or email (optional)]

 📜 License

This project is licensed under the MIT License — feel free to use and improve it!


