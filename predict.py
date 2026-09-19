import pickle
import pandas as pd

# Load trained model
with open("football_model.pkl", "rb") as file:
    model = pickle.load(file)

print("\n⚽ FOOTBALL PLAYER GOALS PREDICTOR")
print("--------------------------------")

# Take player information
age = float(input("Enter player age: "))
mp = float(input("Enter matches played: "))
starts = float(input("Enter starts: "))
minutes = float(input("Enter minutes played: "))
nineties = float(input("Enter 90s played: "))
shots = float(input("Enter shots: "))
shots_on_target = float(input("Enter shots on target: "))
assists = float(input("Enter assists: "))
penalty_goals = float(input("Enter penalty goals: "))
penalty_attempts = float(input("Enter penalty attempts: "))
offsides = float(input("Enter offsides: "))
fouled = float(input("Enter times fouled: "))
crosses = float(input("Enter crosses: "))
fouls = float(input("Enter fouls committed: "))
on_goals = float(input("Enter goals while on pitch (onG): "))
on_goals_against = float(input("Enter goals against while on pitch (onGA): "))
plus_minus = float(input("Enter plus/minus: "))
plus_minus_90 = float(input("Enter plus/minus per 90: "))
on_off = float(input("Enter On-Off: "))

# Create input data
data = pd.DataFrame([[
    age,
    mp,
    starts,
    minutes,
    nineties,
    shots,
    shots_on_target,
    assists,
    penalty_goals,
    penalty_attempts,
    offsides,
    fouled,
    crosses,
    fouls,
    on_goals,
    on_goals_against,
    plus_minus,
    plus_minus_90,
    on_off
]], columns=[
    "Age",
    "MP",
    "Starts",
    "Min",
    "90s",
    "Sh",
    "SoT",
    "Ast",
    "PK",
    "PKatt",
    "Off",
    "Fld",
    "Crs",
    "Fls",
    "onG",
    "onGA",
    "+/-",
    "+/-90",
    "On-Off"
])

# Make prediction
prediction = model.predict(data)[0]

# Prevent negative goal prediction
prediction = max(0, prediction)

print("\n⚽ Predicted Goals:", round(prediction, 2))