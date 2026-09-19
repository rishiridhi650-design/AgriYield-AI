import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# AGRIYIELD AI - EXPLORATORY DATA ANALYSIS
# ==========================================

# Load cleaned dataset
file_path = "data/processed_crop_yield.csv"
df = pd.read_csv(file_path)

print("=" * 55)
print("AGRIYIELD AI - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# ------------------------------------------
# 1. Basic information
# ------------------------------------------

print("\n1. DATASET INFORMATION")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ------------------------------------------
# 2. Most common crops
# ------------------------------------------

print("\n2. TOP 10 CROPS BY NUMBER OF RECORDS")
print(df["Crop"].value_counts().head(10))

# ------------------------------------------
# 3. States
# ------------------------------------------

print("\n3. TOP 10 STATES BY NUMBER OF RECORDS")
print(df["State"].value_counts().head(10))

# ------------------------------------------
# 4. Average yield by crop
# ------------------------------------------

print("\n4. TOP 10 CROPS BY AVERAGE YIELD")

crop_yield = (
    df.groupby("Crop")["Yield"]
    .mean()
    .sort_values(ascending=False)
)

print(crop_yield.head(10))

# ------------------------------------------
# 5. Average yield by season
# ------------------------------------------

print("\n5. AVERAGE YIELD BY SEASON")

season_yield = (
    df.groupby("Season")["Yield"]
    .mean()
    .sort_values(ascending=False)
)

print(season_yield)

# ------------------------------------------
# 6. Average yield by year
# ------------------------------------------

print("\n6. AVERAGE YIELD BY YEAR")

year_yield = df.groupby("Crop_Year")["Yield"].mean()

print(year_yield)

# ------------------------------------------
# 7. Correlation between numerical variables
# ------------------------------------------

print("\n7. NUMERICAL CORRELATION")

numeric_columns = [
    "Crop_Year",
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

correlation = df[numeric_columns].corr()

print(correlation["Yield"].sort_values(ascending=False))

# ------------------------------------------
# 8. Save correlation heatmap
# ------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Agricultural Variables")
plt.tight_layout()

plt.savefig("correlation_heatmap.png", dpi=300)
plt.close()

print("\nCorrelation heatmap saved as: correlation_heatmap.png")

# ------------------------------------------
# 9. Yield distribution
# ------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(df["Yield"], bins=50)

plt.xlabel("Yield")
plt.ylabel("Number of Records")
plt.title("Distribution of Crop Yield")

plt.tight_layout()

plt.savefig("yield_distribution.png", dpi=300)
plt.close()

print("Yield distribution chart saved as: yield_distribution.png")

# ------------------------------------------
# 10. Rainfall vs Yield
# ------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual_Rainfall"],
    df["Yield"],
    alpha=0.3
)

plt.xlabel("Annual Rainfall")
plt.ylabel("Yield")
plt.title("Annual Rainfall vs Crop Yield")

plt.tight_layout()

plt.savefig("rainfall_vs_yield.png", dpi=300)
plt.close()

print("Rainfall vs Yield chart saved as: rainfall_vs_yield.png")

print("\n" + "=" * 55)
print("EDA COMPLETE")
print("=" * 55)