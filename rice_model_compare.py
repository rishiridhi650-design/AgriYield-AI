import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)


# ==================================================
# AGRIYIELD AI - RICE MODEL COMPARISON
# ==================================================

df = pd.read_csv("data/processed_crop_yield.csv")


# --------------------------------------------------
# 1. Keep only Rice
# --------------------------------------------------

rice = df[df["Crop"] == "Rice"].copy()

print("=" * 65)
print("AGRIYIELD AI - RICE YIELD PREDICTION")
print("=" * 65)

print("\nTotal Rice records:", len(rice))

print(
    "Years:",
    rice["Crop_Year"].min(),
    "to",
    rice["Crop_Year"].max()
)


# --------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------

# Avoid division by zero
rice = rice[rice["Area"] > 0].copy()

# Convert total input usage into usage per unit area
rice["Fertilizer_per_area"] = (
    rice["Fertilizer"] / rice["Area"]
)

rice["Pesticide_per_area"] = (
    rice["Pesticide"] / rice["Area"]
)


# --------------------------------------------------
# 3. Select features
# --------------------------------------------------

features = [
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer_per_area",
    "Pesticide_per_area"
]

X = rice[features]

y = rice["Yield"]


# --------------------------------------------------
# 4. Time-based train/test split
# --------------------------------------------------

train_mask = rice["Crop_Year"] <= 2017
test_mask = rice["Crop_Year"] >= 2018

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]


print("\nTraining years: 1997 to 2017")
print("Testing years : 2018 to 2020")

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# --------------------------------------------------
# 5. Feature types
# --------------------------------------------------

categorical_features = [
    "Season",
    "State"
]

numerical_features = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer_per_area",
    "Pesticide_per_area"
]


# --------------------------------------------------
# Function to create preprocessing
# --------------------------------------------------

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features
            )
        ]
    )


# --------------------------------------------------
# 6. Models to compare
# --------------------------------------------------

models = {
    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )
}


results = []


# --------------------------------------------------
# 7. Train and evaluate each model
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "-" * 65)
    print("Training:", model_name)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    median_error = median_absolute_error(
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

    results.append({
        "Model": model_name,
        "MAE": mae,
        "Median_Error": median_error,
        "RMSE": rmse,
        "R2": r2
    })

    print("MAE         :", round(mae, 4))
    print("Median Error:", round(median_error, 4))
    print("RMSE        :", round(rmse, 4))
    print("R2          :", round(r2, 4))


# --------------------------------------------------
# 8. Compare models
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="MAE"
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    comparison.to_string(
        index=False
    )
)


# --------------------------------------------------
# 9. Save comparison
# --------------------------------------------------

comparison.to_csv(
    "rice_model_comparison.csv",
    index=False
)

print(
    "\nSaved results to:"
    " rice_model_comparison.csv"
)

print("\n" + "=" * 65)
print("RICE MODEL COMPARISON COMPLETE")
print("=" * 65)