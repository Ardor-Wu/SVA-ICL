import pandas as pd

df = pd.read_csv("similarity_analysis.csv")
similarity_bins = [0, 0.2, 0.5, 1.0]
df["similarity_group"] = pd.cut(df["similarity_score"], bins=similarity_bins, labels=["0-0.2", "0.2-0.5", "0.5-1"], right=False)

grouped_results = df.groupby("similarity_group")[["accuracy", "f1_score", "mcc"]].mean().round(4)
grouped_results.to_csv("similarity_impact_analysis.csv")