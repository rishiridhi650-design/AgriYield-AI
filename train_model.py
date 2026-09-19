import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


# ==========================================
# AGRIYIELD AI - FIRST MACHINE LEARNING MODEL
# ==========================================


# ------------------------------------------
# 1. Load cleaned dataset
# ------------------------------------------

df = pd.read_csv("data/processed_crop_yield.csv")

print("=" * 60)
print("AGRIYIELD AI - MODEL TRAINING")
print("=" * 60)

print("\nDataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ------------------------------------------
# 2. Select input features
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


# ------------------------------------------
# 3. Select target
# ------------------------------------------

y = df["Yield"]


print("\nInput features:")
for feature in features:
    print("-", feature)

print("\nTarget:")
print("- Yield")


# ------------------------------------------
# 4. Identify categorical columns
# ------------------------------------------

categorical_features = [
    "Crop",
    "Season",
    "State"
]


# ------------------------------------------
# 5. Identify numerical columns
# ------------------------------------------

numerical_features = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]


# ------------------------------------------
# 6. Convert text categories into numbers
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
# 7. Create Random Forest model
# ------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------
# 8. Create complete ML pipeline
# ------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ------------------------------------------
# 9. Split dataset
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------
# 10. Train model
# ------------------------------------------

print("\nTraining Random Forest model...")
print("Please wait...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# ------------------------------------------
# 11. Make predictions
# ------------------------------------------

predictions = pipeline.predict(X_test)


# ------------------------------------------
# 12. Evaluate model
# ------------------------------------------

mae = mean_absolute_error(
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
print("MODEL PERFORMANCE")
print("=" * 60)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)


# ------------------------------------------
# 13. Show sample predictions
# ------------------------------------------

results = X_test.copy()

results["Actual_Yield"] = y_test.values

results["Predicted_Yield"] = predictions

print("\nSAMPLE PREDICTIONS")

print(
    results[
        [
            "Crop",
            "State",
            "Season",
            "Actual_Yield",
            "Predicted_Yield"
        ]
    ].head(15)
)


print("\n" + "=" * 60)
print("FIRST AGRIYIELD AI MODEL COMPLETE")
print("=" * 60)