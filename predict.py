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
shots = float(input("Enter shots: "))
shots_on_target = float(input("Enter shots on target: "))
assists = float(input("Enter assists: "))

# Create input data
data = pd.DataFrame([[
    age,
    mp,
    starts,
    minutes,
    shots,
    shots_on_target,
    assists
]], columns=[
    "Age",
    "MP",
    "Starts",
    "Min",
    "Sh",
    "SoT",
    "Ast"
])

# Make prediction
prediction = model.predict(data)

print("\nPredicted Goals:", round(prediction[0], 2))