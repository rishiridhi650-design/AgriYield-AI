import pandas as pd
import joblib

print("=" * 60)
print("          AGRIYIELD AI - YIELD PREDICTION")
print("=" * 60)

model = joblib.load("models/best_model.pkl")

sample = pd.DataFrame({
    "Crop": ["Rice"],
    "Crop_Year": [2025],
    "Season": ["Kharif"],
    "State": ["Karnataka"],
    "Area": [1000.0],
    "Annual_Rainfall": [1200.0],
    "Fertilizer": [150000.0],
    "Pesticide": [500.0]
})

prediction = model.predict(sample)[0]

print("\nINPUT PARAMETERS")
print("-" * 30)
print(sample.to_string(index=False))

print("\nPREDICTED CROP YIELD")
print("-" * 30)
print(f"Predicted Yield: {prediction:.4f}")

print("\nPrediction completed successfully.")