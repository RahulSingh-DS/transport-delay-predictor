# 🚌 Transit Delay Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Predict public transport (bus & train) delays using weather, time, and event data — powered by LightGBM and deployed via Streamlit.

---

## 🎯 Live Demo

🔗 *Add your Streamlit link here after deployment*

---

## 📌 Overview

Public transport delays are influenced by multiple real-world factors — weather, peak hours, nearby events, and route type. This project builds an **end-to-end ML pipeline** that predicts delay in minutes, helping commuters and transit planners make better decisions.

---

## 🔍 Key Features

- Real-time delay prediction from user inputs
- Interactive Streamlit UI with sidebar controls
- Color-coded results (On Time / Minor Delay / Major Delay)
- Feature importance visualization
- Handles both Bus and Train vehicle types

---

## 📁 Project Structure

```
transit-delay-predictor/
│
├── app.py                   # Streamlit web app
├── requirements.txt         # Dependencies
├── README.md
│
├── notebook/
│   ├── Public Transport Delay Prediction System.ipynb
│   └── model/               # Trained artifacts
│       ├── lgbm_model.pkl
│       ├── label_encoders.pkl
│       └── feature_columns.pkl
│
└── data/
    └── dataset.csv          # Kaggle dataset
```

---

## 📊 Dataset

Synthetic dataset inspired by real-world public transport systems.

| Feature | Description |
|---|---|
| `hour`, `day_of_week`, `month` | Time-based features |
| `temperature`, `wind_speed`, `visibility` | Weather conditions |
| `weather_condition` | Clear / Rain / Fog / Snow / Windy |
| `vehicle_type` | Bus or Train |
| `route_type` | Urban / Suburban / Rural |
| `is_event_nearby` | Binary flag for nearby events |
| `delay_minutes` | **Target** — delay in minutes |

---

## ⚙️ Feature Engineering

- Peak hour indicator (7–9 AM, 5–7 PM)
- Weekend flag
- Label encoding for categorical features
- Feature alignment for consistent model input

---

## 🧠 Model

| Detail | Value |
|---|---|
| Algorithm | LightGBM Regressor |
| Train / Test Split | 80% / 20% |
| Early Stopping | 50 rounds |
| Key Hyperparameters | `n_estimators=500`, `lr=0.05`, `num_leaves=63` |

**Evaluation Metrics:** MAE · RMSE · R²

---

## 🚀 Run Locally

```bash
git clone https://github.com/RahulSingh-DS/transit-delay-predictor.git
cd transit-delay-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py` → Deploy

---

## 🛠 Tech Stack

Python · Pandas · NumPy · LightGBM · Scikit-learn · Streamlit

---

## 👤 Author

**Rahul Singh**
- 💻 [GitHub](https://github.com/RahulSingh-DS)
- 🔗 [LinkedIn](https://linkedin.com/in/pyrahul)

---

## 📄 License

MIT License
