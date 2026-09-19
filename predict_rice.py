import joblib
import pandas as pd


# ==================================================
# AGRIYIELD AI - RICE YIELD PREDICTOR
# ==================================================


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load(
    "models/rice_yield_model.pkl"
)

metadata = joblib.load(
    "models/rice_model_metadata.pkl"
)


print("=" * 60)
print("          AGRIYIELD AI")
print("      RICE YIELD PREDICTOR")
print("=" * 60)


# --------------------------------------------------
# Show available seasons
# --------------------------------------------------

print("\nAvailable seasons:")

for season in metadata["seasons"]:
    print("-", season)


# --------------------------------------------------
# User input
# --------------------------------------------------

year = int(
    input(
        "\nEnter crop year: "
    )
)

season = input(
    "Enter season: "
).strip()

state = input(
    "Enter state: "
).strip()

area = float(
    input(
        "Enter cultivated area: "
    )
)

rainfall = float(
    input(
        "Enter annual rainfall: "
    )
)

fertilizer = float(
    input(
        "Enter fertilizer value: "
    )
)

pesticide = float(
    input(
        "Enter pesticide value: "
    )
)


# --------------------------------------------------
# Validate basic values
# --------------------------------------------------

if area <= 0:
    print(
        "\nERROR: Area must be greater than zero."
    )
    raise SystemExit


if rainfall < 0:
    print(
        "\nERROR: Rainfall cannot be negative."
    )
    raise SystemExit


if fertilizer < 0:
    print(
        "\nERROR: Fertilizer cannot be negative."
    )
    raise SystemExit


if pesticide < 0:
    print(
        "\nERROR: Pesticide cannot be negative."
    )
    raise SystemExit


# --------------------------------------------------
# Warn about unknown categories
# --------------------------------------------------

if state not in metadata["states"]:
    print(
        "\nWARNING:"
        " This state was not present"
        " in the Rice training data."
    )


if season not in metadata["seasons"]:
    print(
        "\nWARNING:"
        " This season was not present"
        " in the Rice training data."
    )


if year > metadata["maximum_year"]:
    print(
        "\nNOTE:"
        " The dataset ends in",
        metadata["maximum_year"],
        "so predictions beyond that year"
        " are experimental."
    )


# --------------------------------------------------
# Create one-row input table
# --------------------------------------------------

input_data = pd.DataFrame(
    [{
        "Crop_Year": year,
        "Season": season,
        "State": state,
        "Area": area,
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }]
)


# --------------------------------------------------
# Make prediction
# --------------------------------------------------

prediction = model.predict(
    input_data
)[0]


print("\n" + "=" * 60)

print(
    "Predicted Rice Yield:",
    round(prediction, 4)
)

print("=" * 60)

print(
    "\nNote: This is an ML estimate based on"
    " historical dataset patterns."
)