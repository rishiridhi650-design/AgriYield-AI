import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# AGRIYIELD AI - FEATURE SET COMPARISON
# ==================================================

df = pd.read_csv("data/processed_crop_yield.csv")

rice = df[df["Crop"] == "Rice"].copy()

rice = rice[rice["Area"] > 0].copy()


# --------------------------------------------------
# Create engineered features
# --------------------------------------------------

rice["Fertilizer_per_area"] = (
    rice["Fertilizer"] / rice["Area"]
)

rice["Pesticide_per_area"] = (
    rice["Pesticide"] / rice["Area"]
)


# --------------------------------------------------
# Target
# --------------------------------------------------

y = rice["Yield"]


# --------------------------------------------------
# Time split
# --------------------------------------------------

train_mask = rice["Crop_Year"] <= 2017
test_mask = rice["Crop_Year"] >= 2018


# --------------------------------------------------
# Two feature sets
# --------------------------------------------------

feature_sets = {

    "RAW FEATURES": [
        "Crop_Year",
        "Season",
        "State",
        "Area",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide"
    ],

    "PER-AREA FEATURES": [
        "Crop_Year",
        "Season",
        "State",
        "Area",
        "Annual_Rainfall",
        "Fertilizer_per_area",
        "Pesticide_per_area"
    ]
}


results = []


# --------------------------------------------------
# Test each feature set
# --------------------------------------------------

for feature_name, features in feature_sets.items():

    print("\n" + "=" * 65)
    print(feature_name)
    print("=" * 65)

    X = rice[features]

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]


    categorical_features = [
        "Season",
        "State"
    ]


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )


    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )


    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


    print("Training...")

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


    print("MAE         :", round(mae, 6))
    print("Median Error:", round(median_error, 6))
    print("RMSE        :", round(rmse, 6))
    print("R2          :", round(r2, 6))


    results.append({
        "Feature_Set": feature_name,
        "MAE": mae,
        "Median_Error": median_error,
        "RMSE": rmse,
        "R2": r2
    })


# --------------------------------------------------
# Final comparison
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "MAE"
)


print("\n" + "=" * 65)
print("FINAL FEATURE COMPARISON")
print("=" * 65)

print(
    comparison.to_string(
        index=False
    )
)


comparison.to_csv(
    "rice_feature_comparison.csv",
    index=False
)


print(
    "\nSaved: rice_feature_comparison.csv"
)