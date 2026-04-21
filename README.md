# 🚌 Transit Delay Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Predict public transport (bus & train) delays using real-world timetable data, weather conditions, and local event flags — powered by LightGBM and deployed via Streamlit.

---

## 📌 Problem Statement

Urban transit systems are heavily affected by weather, traffic, and local events. This project builds a machine learning pipeline to **predict delay in minutes** for buses and trains, enabling commuters and transit authorities to plan better.

---

## 🎯 Live Demo

> 🔗 [Click here to try the app](https://your-app-name.streamlit.app) *(deploy to Streamlit Cloud and update this link)*

![App Screenshot](assets/screenshot.png)

---

## 📁 Project Structure

```
transit-delay-prediction/
│
├── app.py                  # Streamlit web application
├── train_model.py          # Data preprocessing + model training
├── requirements.txt        # Python dependencies
├── dataset.csv             # Raw dataset (Kaggle)
│
├── model/
│   ├── lgbm_model.pkl      # Trained LightGBM model
│   ├── label_encoders.pkl  # Saved label encoders
│   └── feature_columns.pkl # Training feature order
│
└── README.md
```

---

## 📊 Dataset

**Source:** [Public Transport Delays with Weather & Events — Kaggle](https://www.kaggle.com)

The dataset simulates public transport delays influenced by:
- 🕐 Timetable data (scheduled vs actual times)
- 🌦 Weather conditions (rain, fog, snow, wind speed, temperature)
- 📍 Nearby events (concerts, sports, etc.)
- 🚌 Vehicle type (Bus / Train)

**Key columns:**

| Column | Description |
|---|---|
| `scheduled_time` | Planned departure datetime |
| `vehicle_type` | Bus or Train |
| `weather_condition` | Clear / Rain / Fog / Snow / Windy |
| `temperature` | Temperature in °C |
| `wind_speed` | Wind speed in km/h |
| `visibility_km` | Visibility in km |
| `is_event_nearby` | Binary flag for nearby events |
| `delay_minutes` | **Target** — actual delay in minutes |

---

## ⚙️ Feature Engineering

Features derived from raw data:

| Feature | Description |
|---|---|
| `hour` | Departure hour (0–23) |
| `day_of_week` | Day number (0=Mon, 6=Sun) |
| `month` | Month of year |
| `is_weekend` | 1 if Saturday/Sunday |
| `is_peak_hour` | 1 if 7–9 AM or 5–7 PM |

---

## 🧠 Model

| Detail | Value |
|---|---|
| Algorithm | LightGBM Regressor |
| Objective | Regression (predict delay in minutes) |
| Train/Test Split | 80% / 20% |
| Early Stopping | Yes (50 rounds) |
| Key Hyperparameters | `n_estimators=500`, `lr=0.05`, `num_leaves=63` |

**Evaluation Metrics:**

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error (minutes) |
| RMSE | Root Mean Square Error |
| R² | Coefficient of Determination |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/RahulSingh-DS/transit-delay-prediction.git
cd transit-delay-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place your CSV file in the root folder and rename it:
```bash
mv your_file.csv dataset.csv
```

### 4. Train the model
```bash
python train_model.py
```
This will generate the `model/` folder with all saved artifacts.

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `app.py` → Deploy
4. Make sure your `model/` folder with `.pkl` files is committed to GitHub

---

## 📈 Results

| Metric | Score |
|---|---|
| MAE | *Run train_model.py to see* |
| RMSE | *Run train_model.py to see* |
| R² | *Run train_model.py to see* |

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **ML:** LightGBM, Scikit-learn
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

---

## 👤 Author

**Rahul Singh**
- 🔗 [LinkedIn](https://linkedin.com/in/pyrahul)
- 💻 [GitHub](https://github.com/RahulSingh-DS)

---

## 📄 License

This project is licensed under the MIT License.
