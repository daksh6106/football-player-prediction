import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("dataset/players_data-2026_2027.csv")

# Features used by the best-performing model
features = [
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
]

target = "Gls"

# Keep required columns and remove missing values
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Best model found during comparison
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_samples_split=4,
    min_samples_leaf=2,
    loss="huber",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test predictions
predictions = model.predict(X_test)

# Metrics
test_r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# 5-fold cross-validation
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

cv_r2 = cv_scores.mean()

print("========================================")
print("FOOTBALL PLAYER PREDICTION MODEL")
print("========================================")
print("Model: Gradient Boosting")
print("Test R2:", round(test_r2, 4))
print("CV R2:", round(cv_r2, 4))
print("MAE:", round(mae, 4))
print("========================================")

# Save trained model
with open("football_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("football_model.pkl saved successfully.")
