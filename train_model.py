import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Load dataset
df = pd.read_csv("dataset/players_data-2026_2027.csv")

# Select useful columns
features = [
    "Age",
    "MP",
    "Starts",
    "Min",
    "Sh",
    "SoT",
    "Ast"
]

target = "Gls"

# Remove rows with missing values
df = df[features + [target]].dropna()

# Input and output
X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Football Player Performance Prediction")
print("---------------------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Save model
with open("football_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as football_model.pkl")