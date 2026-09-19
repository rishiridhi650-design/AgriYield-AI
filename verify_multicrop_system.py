import joblib
import pandas as pd

print("=" * 75)
print("AGRIYIELD AI - MULTI-CROP SYSTEM VERIFICATION")
print("=" * 75)

# Load dataset
df = pd.read_csv("data/processed_crop_yield.csv")

# Load metadata
metadata = joblib.load(
    "models/multicrop/multicrop_metadata.pkl"
)

features = [
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

results = []

for crop, info in metadata.items():

    # Skip weak models
    if info["test_r2"] <= 0:
        print(
            f"\nSKIPPED: {crop} "
            f"(Temporal R2 = {info['test_r2']:.4f})"
        )
        continue

    print("\n" + "-" * 75)
    print("Testing crop:", crop)

    # Load saved crop model
    model = joblib.load(
        info["model_path"]
    )

    # Get crop data
    crop_df = df[
        df["Crop"] == crop
    ].copy()

    # Pick latest historical row
    sample = (
        crop_df
        .sort_values("Crop_Year", ascending=False)
        .iloc[0]
    )

    input_data = pd.DataFrame(
        [{
            "Crop_Year": sample["Crop_Year"],
            "Season": sample["Season"],
            "State": sample["State"],
            "Area": sample["Area"],
            "Annual_Rainfall": sample["Annual_Rainfall"],
            "Fertilizer": sample["Fertilizer"],
            "Pesticide": sample["Pesticide"]
        }]
    )

    prediction = model.predict(
        input_data
    )[0]

    actual = sample["Yield"]

    absolute_error = abs(
        actual - prediction
    )

    print("Model:", info["best_algorithm"])
    print("Year:", int(sample["Crop_Year"]))
    print("State:", sample["State"])
    print("Actual Yield:", round(actual, 4))
    print("Predicted Yield:", round(prediction, 4))
    print("Absolute Difference:", round(absolute_error, 4))
    print("Temporal Test R2:", round(info["test_r2"], 4))

    results.append({
        "Crop": crop,
        "Model": info["best_algorithm"],
        "Year": int(sample["Crop_Year"]),
        "Actual_Yield": actual,
        "Predicted_Yield": prediction,
        "Absolute_Difference": absolute_error,
        "Temporal_Test_MAE": info["test_mae"],
        "Temporal_Test_R2": info["test_r2"]
    })

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "multicrop_verification_results.csv",
    index=False
)

print("\n")
print("=" * 75)
print("SYSTEM VERIFICATION COMPLETE")
print("=" * 75)

print(
    "\nSuccessfully verified models:",
    len(results_df)
)

print(
    "\nResults saved to:"
)

print(
    "multicrop_verification_results.csv"
)

print(
    "\nNote:"
)

print(
    "This is only a functionality check."
)

print(
    "Use the earlier 2018-2020 temporal test metrics "
    "for actual model evaluation."
)