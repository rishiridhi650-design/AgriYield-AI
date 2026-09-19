import os
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)


# ==========================================================
# AGRIYIELD AI - MULTI-CROP MODEL TRAINING
# ==========================================================

print("=" * 70)
print("AGRIYIELD AI - MULTI-CROP MODEL TRAINING")
print("=" * 70)


# ----------------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------------

df = pd.read_csv(
    "data/processed_crop_yield.csv"
)


# ----------------------------------------------------------
# 2. Select top 10 crops by number of records
# ----------------------------------------------------------

top_crops = (
    df["Crop"]
    .value_counts()
    .head(10)
    .index
    .tolist()
)

print("\nCrops selected:")

for crop in top_crops:
    print("-", crop)


# ----------------------------------------------------------
# 3. Input features
# ----------------------------------------------------------

features = [
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]


categorical_features = [
    "Season",
    "State"
]


numerical_features = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]


# ----------------------------------------------------------
# 4. Preprocessor function
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# 5. Model function
# ----------------------------------------------------------

def create_model(model_name):

    if model_name == "Linear Regression":

        return LinearRegression()

    elif model_name == "Random Forest":

        return RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "Gradient Boosting":

        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )


# ----------------------------------------------------------
# 6. Create model directory
# ----------------------------------------------------------

os.makedirs(
    "models/multicrop",
    exist_ok=True
)


# ----------------------------------------------------------
# 7. Storage for metadata and results
# ----------------------------------------------------------

metadata = {}

all_results = []


# ==========================================================
# 8. TRAIN MODELS FOR EACH CROP
# ==========================================================

for crop in top_crops:

    print("\n")
    print("=" * 70)
    print("CROP:", crop)
    print("=" * 70)


    crop_df = df[
        df["Crop"] == crop
    ].copy()


    crop_df = crop_df[
        crop_df["Area"] > 0
    ].copy()


    print(
        "Total records:",
        len(crop_df)
    )


    # ------------------------------------------------------
    # 9. Time-based split
    # ------------------------------------------------------

    train_df = crop_df[
        crop_df["Crop_Year"] <= 2017
    ].copy()

    test_df = crop_df[
        crop_df["Crop_Year"] >= 2018
    ].copy()


    print(
        "Training records:",
        len(train_df)
    )

    print(
        "Testing records:",
        len(test_df)
    )


    # Skip if there are too few recent records
    if len(test_df) < 10:

        print(
            "Not enough recent test data."
        )

        print(
            "Skipping crop:",
            crop
        )

        continue


    X_train = train_df[features]
    y_train = train_df["Yield"]

    X_test = test_df[features]
    y_test = test_df["Yield"]


    # ------------------------------------------------------
    # 10. Models to compare
    # ------------------------------------------------------

    model_names = [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ]


    crop_results = []


    # ------------------------------------------------------
    # 11. Train and evaluate each model
    # ------------------------------------------------------

    for model_name in model_names:

        print(
            "\nTraining:",
            model_name
        )


        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor()
                ),
                (
                    "model",
                    create_model(
                        model_name
                    )
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


        print(
            "MAE:",
            round(mae, 4)
        )

        print(
            "Median Error:",
            round(median_error, 4)
        )

        print(
            "RMSE:",
            round(rmse, 4)
        )

        print(
            "R2:",
            round(r2, 4)
        )


        crop_results.append({
            "Crop": crop,
            "Model": model_name,
            "MAE": mae,
            "Median_Error": median_error,
            "RMSE": rmse,
            "R2": r2
        })


    # ------------------------------------------------------
    # 12. Select best model based on lowest MAE
    # ------------------------------------------------------

    crop_results_df = pd.DataFrame(
        crop_results
    )


    crop_results_df = (
        crop_results_df
        .sort_values("MAE")
    )


    best = crop_results_df.iloc[0]

    best_model_name = best["Model"]


    print("\n" + "-" * 70)

    print(
        "SELECTED MODEL:",
        best_model_name
    )

    print(
        "Test MAE:",
        round(best["MAE"], 4)
    )

    print(
        "Test RMSE:",
        round(best["RMSE"], 4)
    )

    print(
        "Test R2:",
        round(best["R2"], 4)
    )

    print("-" * 70)


    # ------------------------------------------------------
    # 13. Train selected model on full crop dataset
    # ------------------------------------------------------

    final_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                create_model(
                    best_model_name
                )
            )
        ]
    )


    X_full = crop_df[features]

    y_full = crop_df["Yield"]


    final_pipeline.fit(
        X_full,
        y_full
    )


    # ------------------------------------------------------
    # 14. Create safe filename
    # ------------------------------------------------------

    safe_crop_name = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        crop
    ).strip("_").lower()


    model_path = (
        f"models/multicrop/"
        f"{safe_crop_name}_model.pkl"
    )


    # ------------------------------------------------------
    # 15. Save model
    # ------------------------------------------------------

    joblib.dump(
        final_pipeline,
        model_path
    )


    print(
        "Saved model:",
        model_path
    )


    # ------------------------------------------------------
    # 16. Save crop metadata
    # ------------------------------------------------------

    metadata[crop] = {

        "model_path":
            model_path,

        "states":
            sorted(
                crop_df["State"]
                .unique()
                .tolist()
            ),

        "seasons":
            sorted(
                crop_df["Season"]
                .unique()
                .tolist()
            ),

        "minimum_year":
            int(
                crop_df[
                    "Crop_Year"
                ].min()
            ),

        "maximum_year":
            int(
                crop_df[
                    "Crop_Year"
                ].max()
            ),

        "best_algorithm":
            best_model_name,

        "test_mae":
            float(best["MAE"]),

        "test_rmse":
            float(best["RMSE"]),

        "test_r2":
            float(best["R2"]),

        "records":
            int(len(crop_df))
    }


    all_results.extend(
        crop_results
    )


# ==========================================================
# 17. SAVE SYSTEM METADATA
# ==========================================================

joblib.dump(
    metadata,
    "models/multicrop/multicrop_metadata.pkl"
)


# ----------------------------------------------------------
# 18. Save evaluation table
# ----------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)


results_df.to_csv(
    "multicrop_model_results.csv",
    index=False
)


# ----------------------------------------------------------
# 19. Final summary
# ----------------------------------------------------------

print("\n")
print("=" * 70)
print("MULTI-CROP TRAINING COMPLETE")
print("=" * 70)


print(
    "\nModels created:",
    len(metadata)
)


print("\nAvailable crops:")


for crop, information in metadata.items():

    print(
        crop,
        "->",
        information["best_algorithm"],
        "| MAE:",
        round(
            information["test_mae"],
            4
        ),
        "| R2:",
        round(
            information["test_r2"],
            4
        )
    )


print(
    "\nSaved metadata:"
)

print(
    "models/multicrop/"
    "multicrop_metadata.pkl"
)


print(
    "\nSaved evaluation table:"
)

print(
    "multicrop_model_results.csv"
)


print("\nAGRIYIELD AI MULTI-CROP SYSTEM READY")