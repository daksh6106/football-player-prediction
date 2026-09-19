# ⚽ Football Player Performance Prediction
## 🚀 Live Demo

[Open Football Player Prediction App](https://football-player-prediction-chupmzlv4zghz8thfteaqp.streamlit.app/)

A simple Machine Learning project that predicts the expected goals of a football player using player statistics.

## 📌 Project Description

This project uses Machine Learning to predict the number of goals a football player may score based on different player statistics.
s
The model uses:

- Age
- Matches Played
- Starts
- Minutes Played
- Shots
- Shots on Target
- Assists

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Regression
- Pickle

## 🤖 Machine Learning Model

The project uses:

**Random Forest Regressor**

Model Performance:

- Mean Absolute Error: 0.24
- R² Score: 0.58

## 📂 Project Structure

```text
FootballPlayerPrediction/
│
├── dataset/
│   └── players_data-2026_2027.csv
│
├── train_model.py
├── predict.py
├── football_model.pkl
├── requirements.txt
└── README.md
