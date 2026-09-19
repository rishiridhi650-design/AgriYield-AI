import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)


# ==========================================
# AGRIYIELD AI - TIME BASED VALIDATION
# ==========================================

df = pd.read_csv("data/processed_crop_yield.csv")

print("=" * 60)
print("AGRIYIELD AI - TIME BASED VALIDATION")
print("=" * 60)


# ------------------------------------------
# Features and target
# ------------------------------------------

features = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

X = df[features]
y = df["Yield"]


# ------------------------------------------
# Train on old years
# Test on newer years
# ------------------------------------------

train_mask = df["Crop_Year"] <= 2017
test_mask = df["Crop_Year"] >= 2018

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print("\nTraining years:")
print(
    df.loc[train_mask, "Crop_Year"].min(),
    "to",
    df.loc[train_mask, "Crop_Year"].max()
)

print("\nTesting years:")
print(
    df.loc[test_mask, "Crop_Year"].min(),
    "to",
    df.loc[test_mask, "Crop_Year"].max()
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------
# Categorical columns
# ------------------------------------------

categorical_features = [
    "Crop",
    "Season",
    "State"
]


# ------------------------------------------
# Preprocessing
# ------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ------------------------------------------
# Random Forest
# ------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ------------------------------------------
# Train
# ------------------------------------------

print("\nTraining model on historical years...")
pipeline.fit(X_train, y_train)

print("Training complete!")


# ------------------------------------------
# Predict
# ------------------------------------------

predictions = pipeline.predict(X_test)


# ------------------------------------------
# Metrics
# ------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

median_ae = median_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 60)
print("TIME BASED MODEL PERFORMANCE")
print("=" * 60)

print("MAE              :", mae)
print("Median Abs Error :", median_ae)
print("RMSE             :", rmse)
print("R2               :", r2)


# ------------------------------------------
# Sample predictions
# ------------------------------------------

results = X_test.copy()

results["Actual_Yield"] = y_test.values
results["Predicted_Yield"] = predictions

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("\nSAMPLE FUTURE-YEAR PREDICTIONS")

print(
    results[
        [
            "Crop",
            "Crop_Year",
            "State",
            "Actual_Yield",
            "Predicted_Yield"
        ]
    ].head(20)
)


print("\n" + "=" * 60)
print("TIME VALIDATION COMPLETE")
print("=" * 60)