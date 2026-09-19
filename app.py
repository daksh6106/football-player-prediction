import streamlit as st
import pandas as pd
import pickle

# ===============================
# Load trained model
# ===============================

with open("football_model.pkl", "rb") as file:
    model = pickle.load(file)


# ===============================
# Page
# ===============================

st.set_page_config(
    page_title="Football Player Goals Predictor",
    page_icon="⚽"
)

st.title("⚽ Football Player Goals Predictor")

st.write(
    "Enter player statistics to predict the expected number of goals."
)


# ===============================
# Player Statistics
# ===============================

st.subheader("📊 Player Statistics")


age = st.number_input(
    "Age",
    min_value=15,
    max_value=50,
    value=25
)

mp = st.number_input(
    "Matches Played",
    min_value=0,
    value=30
)

starts = st.number_input(
    "Starts",
    min_value=0,
    value=25
)

minutes = st.number_input(
    "Minutes Played",
    min_value=0,
    value=2200
)

nineties = st.number_input(
    "90s Played",
    min_value=0.0,
    value=24.0
)

shots = st.number_input(
    "Shots",
    min_value=0,
    value=50
)

shots_on_target = st.number_input(
    "Shots on Target",
    min_value=0,
    value=20
)

assists = st.number_input(
    "Assists",
    min_value=0,
    value=5
)

penalty_goals = st.number_input(
    "Penalty Goals",
    min_value=0,
    value=0
)

penalty_attempts = st.number_input(
    "Penalty Attempts",
    min_value=0,
    value=0
)

offsides = st.number_input(
    "Offsides",
    min_value=0,
    value=2
)

fouled = st.number_input(
    "Times Fouled",
    min_value=0,
    value=10
)

crosses = st.number_input(
    "Crosses",
    min_value=0,
    value=20
)

fouls = st.number_input(
    "Fouls Committed",
    min_value=0,
    value=10
)

on_goals = st.number_input(
    "Goals While On Pitch (onG)",
    min_value=0,
    value=30
)

on_goals_against = st.number_input(
    "Goals Against While On Pitch (onGA)",
    min_value=0,
    value=20
)

plus_minus = st.number_input(
    "Plus/Minus",
    value=5
)

plus_minus_90 = st.number_input(
    "Plus/Minus per 90",
    value=0.2
)

on_off = st.number_input(
    "On-Off",
    value=0.1
)


# ===============================
# Prediction
# ===============================

if st.button("⚽ Predict Goals"):

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

    prediction = model.predict(data)[0]

    # Avoid negative goal predictions
    prediction = max(0, prediction)

    st.success(
        f"⚽ Predicted Goals: {prediction:.2f}"
    )

    st.info(
        "Prediction generated using the Gradient Boosting model."
    )