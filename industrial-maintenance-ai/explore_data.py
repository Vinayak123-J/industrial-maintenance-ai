import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
df.info()

# Failure distribution
target = "Machine failure"

if target in df.columns:
    print("\nMachine failure counts:")
    print(df[target].value_counts())

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=target)
    plt.title("Machine Failure Distribution")
    plt.xlabel("Machine failure (0 = No, 1 = Yes)")
    plt.ylabel("Number of observations")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "failure_distribution.png")
    plt.close()

    print("\nSaved graph to reports/failure_distribution.png")
else:
    print("\nTarget column not found. Check the CSV column names.")

print("\nData exploration completed!")