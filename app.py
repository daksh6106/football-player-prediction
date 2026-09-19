import streamlit as st
import pandas as pd
import pickle

# Load model
with open("football_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("⚽ Football Player Goals Predictor")
st.write("Enter player statistics to predict expected goals.")

age = st.number_input("Age", min_value=15, max_value=50, value=25)
mp = st.number_input("Matches Played", min_value=0, value=30)
starts = st.number_input("Starts", min_value=0, value=25)
minutes = st.number_input("Minutes Played", min_value=0, value=2200)
shots = st.number_input("Shots", min_value=0, value=50)
shots_on_target = st.number_input("Shots on Target", min_value=0, value=20)
assists = st.number_input("Assists", min_value=0, value=5)

if st.button("Predict Goals"):
    data = pd.DataFrame([[
        age, mp, starts, minutes, shots, shots_on_target, assists
    ]], columns=[
        "Age", "MP", "Starts", "Min", "Sh", "SoT", "Ast"
    ])

    prediction = model.predict(data)[0]

    st.success(f"Predicted Goals: {prediction:.2f}")