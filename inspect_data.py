import pandas as pd

# Load dataset
file_path = "data/raw/crop_yield.csv"
df = pd.read_csv(file_path)

print("=" * 50)
print("AGRIYIELD AI - DATASET INSPECTION")
print("=" * 50)

# Dataset size
print("\n1. DATASET SIZE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Column names
print("\n2. COLUMNS")
for column in df.columns:
    print("-", column)

# Unique values
print("\n3. UNIQUE VALUES")

print("\nCrops:")
print(df["Crop"].nunique())

print("\nSeasons:")
print(df["Season"].unique())

print("\nStates:")
print(df["State"].nunique())

print("\nYears:")
print(df["Crop_Year"].min(), "to", df["Crop_Year"].max())

# Basic statistics
print("\n4. NUMERICAL STATISTICS")
print(df.describe())

# Target information
print("\n5. YIELD INFORMATION")
print("Minimum Yield:", df["Yield"].min())
print("Maximum Yield:", df["Yield"].max())
print("Average Yield:", df["Yield"].mean())

print("\n" + "=" * 50)
print("INSPECTION COMPLETE")
print("=" * 50)