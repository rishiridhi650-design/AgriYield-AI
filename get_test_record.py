import pandas as pd

df = pd.read_csv("data/processed_crop_yield.csv")

rice = df[
    (df["Crop"] == "Rice") &
    (df["Crop_Year"] == 2020)
].copy()

sample = rice.sample(
    n=1,
    random_state=42
)

columns = [
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

print("\nTEST RECORD")
print("=" * 50)

for column in columns:
    print(f"{column}: {sample.iloc[0][column]}")

print("=" * 50)
print("Yield is the actual historical answer.")