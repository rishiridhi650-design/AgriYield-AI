import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ==========================================
# AGRIYIELD AI - ERROR ANALYSIS
# ==========================================

df = pd.read_csv("data/processed_crop_yield.csv")


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
# Time based split
# ------------------------------------------

train_mask = df["Crop_Year"] <= 2017
test_mask = df["Crop_Year"] >= 2018


X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]


# ------------------------------------------
# Preprocessing
# ------------------------------------------

categorical_features = [
    "Crop",
    "Season",
    "State"
]


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
# Model
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


print("Training model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete!")


# ------------------------------------------
# Predictions
# ------------------------------------------

predictions = pipeline.predict(X_test)


results = X_test.copy()

results["Actual_Yield"] = y_test.values

results["Predicted_Yield"] = predictions

results["Absolute_Error"] = np.abs(
    results["Actual_Yield"]
    - results["Predicted_Yield"]
)


# ==========================================
# 1. Biggest individual errors
# ==========================================

print("\n" + "=" * 65)
print("TOP 20 BIGGEST PREDICTION ERRORS")
print("=" * 65)

largest_errors = results.sort_values(
    "Absolute_Error",
    ascending=False
)

print(
    largest_errors[
        [
            "Crop",
            "Crop_Year",
            "State",
            "Actual_Yield",
            "Predicted_Yield",
            "Absolute_Error"
        ]
    ].head(20)
)


# ==========================================
# 2. Average error by crop
# ==========================================

crop_errors = (
    results
    .groupby("Crop")
    .agg(
        Records=("Actual_Yield", "count"),
        Average_Actual_Yield=("Actual_Yield", "mean"),
        MAE=("Absolute_Error", "mean"),
        Median_Error=("Absolute_Error", "median")
    )
    .sort_values(
        "MAE",
        ascending=False
    )
)


print("\n" + "=" * 65)
print("CROPS WITH HIGHEST AVERAGE ERROR")
print("=" * 65)

print(
    crop_errors.head(20)
)


# ==========================================
# 3. Crops with lowest error
# ==========================================

print("\n" + "=" * 65)
print("CROPS WITH LOWEST AVERAGE ERROR")
print("=" * 65)

print(
    crop_errors.sort_values(
        "MAE"
    ).head(20)
)


# ==========================================
# Save results
# ==========================================

crop_errors.to_csv(
    "crop_error_analysis.csv"
)

largest_errors.to_csv(
    "prediction_errors.csv",
    index=False
)


print("\nFiles saved:")
print("- crop_error_analysis.csv")
print("- prediction_errors.csv")

print("\nERROR ANALYSIS COMPLETE")