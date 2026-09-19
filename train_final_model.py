import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ==================================================
# AGRIYIELD AI - FINAL RICE MODEL
# ==================================================

print("=" * 60)
print("AGRIYIELD AI - TRAIN FINAL MODEL")
print("=" * 60)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed_crop_yield.csv"
)


# --------------------------------------------------
# 2. Keep only Rice records
# --------------------------------------------------

rice = df[
    df["Crop"] == "Rice"
].copy()


print("\nRice records:", len(rice))

print(
    "Years:",
    rice["Crop_Year"].min(),
    "to",
    rice["Crop_Year"].max()
)


# --------------------------------------------------
# 3. Select chosen raw features
# --------------------------------------------------

features = [
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]


X = rice[features]

y = rice["Yield"]


# --------------------------------------------------
# 4. Define categorical features
# --------------------------------------------------

categorical_features = [
    "Season",
    "State"
]


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

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


# --------------------------------------------------
# 6. Random Forest
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 7. Complete pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# --------------------------------------------------
# 8. Train using all available Rice data
# --------------------------------------------------

print("\nTraining final model...")

pipeline.fit(
    X,
    y
)

print("Training complete!")


# --------------------------------------------------
# 9. Create model directory
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

model_path = (
    "models/rice_yield_model.pkl"
)

joblib.dump(
    pipeline,
    model_path
)


print(
    "\nModel saved successfully:"
)

print(model_path)


# --------------------------------------------------
# 11. Save allowed categories for later use
# --------------------------------------------------

states = sorted(
    rice["State"].unique()
)

seasons = sorted(
    rice["Season"].unique()
)


metadata = {
    "states": states,
    "seasons": seasons,
    "minimum_year":
        int(rice["Crop_Year"].min()),
    "maximum_year":
        int(rice["Crop_Year"].max())
}


joblib.dump(
    metadata,
    "models/rice_model_metadata.pkl"
)


print(
    "Metadata saved:"
    " models/rice_model_metadata.pkl"
)


print("\n" + "=" * 60)
print("FINAL MODEL READY")
print("=" * 60)