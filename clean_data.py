import pandas as pd

# ==========================================
# AGRIYIELD AI - DATA CLEANING
# ==========================================

# Load original dataset
input_file = "data/raw/crop_yield.csv"

df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# ------------------------------------------
# 1. Remove extra spaces from column names
# ------------------------------------------

df.columns = df.columns.str.strip()

# ------------------------------------------
# 2. Remove extra spaces from text columns
# ------------------------------------------

text_columns = ["Crop", "Season", "State"]

for column in text_columns:
    df[column] = df[column].astype(str).str.strip()

# ------------------------------------------
# 3. Check missing values
# ------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------------------
# 4. Check duplicate rows
# ------------------------------------------

print("\nDuplicate rows:", df.duplicated().sum())

# ------------------------------------------
# 5. Remove duplicate rows if any
# ------------------------------------------

df = df.drop_duplicates()

# ------------------------------------------
# 6. Display cleaned categories
# ------------------------------------------

print("\nSeasons after cleaning:")
print(df["Season"].unique())

print("\nNumber of crops:", df["Crop"].nunique())
print("Number of states:", df["State"].nunique())

# ------------------------------------------
# 7. Save cleaned dataset
# ------------------------------------------

output_file = "data/processed_crop_yield.csv"

df.to_csv(output_file, index=False)

print("\nCleaned dataset shape:", df.shape)
print("Cleaned dataset saved to:", output_file)

print("\n==========================================")
print("DATA CLEANING COMPLETE")
print("==========================================")