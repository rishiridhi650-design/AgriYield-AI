import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# AGRIYIELD AI - ML MODEL TRAINING
# ==========================================

print("=" * 60)
print("           AGRIYIELD AI - MODEL TRAINING")
print("=" * 60)

# 1. Load dataset
df = pd.read_csv("data/raw/crop_yield.csv")

print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# 2. Remove target leakage
# Production is excluded because it is directly related to Yield.
df = df.drop(columns=["Production"])


# 3. Separate features and target
X = df.drop(columns=["Yield"])
y = df["Yield"]


# 4. Identify column types
categorical_features = ["Crop", "Season", "State"]
numerical_features = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]


# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_features
        )
    ]
)


# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# 7. Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}


# 8. Train and evaluate
results = []
trained_models = {}

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = pipeline

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")


# 9. Compare models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# 10. Select best model
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model_name}")
print("=" * 60)


# 11. Save results
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# 12. Save best model
joblib.dump(
    best_model,
    "models/best_model.pkl"
)


# 13. Save a sample prediction
sample = X_test.iloc[[0]]
actual = y_test.iloc[0]
predicted = best_model.predict(sample)[0]

print("\nSample Prediction")
print("-" * 30)
print(f"Actual Yield   : {actual:.4f}")
print(f"Predicted Yield: {predicted:.4f}")


print("\nModel saved to: models/best_model.pkl")
print("Results saved to: results/model_comparison.csv")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)