import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("Compound_OBP_binding.csv")
obp_cols = df.columns[2:]

# Number of VOCs measured per OBP (this is the "n_competitors" used to compute s5)
n_per_obp = df[obp_cols].notna().sum()

print(f"Total OBPs: {len(obp_cols)}")
print(f"Total VOCs: {len(df)}")
print(f"Median: {n_per_obp.median():.0f}")
print(f"Mean: {n_per_obp.mean():.1f}")
print(f"Standard deviation: {n_per_obp.std():.1f}")
print(f"Variance: {n_per_obp.var():.1f}")
print(f"Mode: {n_per_obp.mode().tolist()}")
print(f"Quartiles: {n_per_obp.quantile([0.25, 0.5, 0.75]).to_dict()}")
print(f"Range: {n_per_obp.min()} - {n_per_obp.max()}")
print(f"Interquartile range (IQR): {n_per_obp.quantile(0.75) - n_per_obp.quantile(0.25):.1f}")
print(f"Skewness: {n_per_obp.skew():.2f}")
print(f"Kurtosis: {n_per_obp.kurt():.2f}")
print(f"OBPs with >= 10 measurements: {(n_per_obp >= 10).sum()} / {len(obp_cols)}")
print(f"OBPs with >= 25 measurements: {(n_per_obp >= 25).sum()} / {len(obp_cols)}")

print("\nFull statistical summary:")
print(n_per_obp.describe())
