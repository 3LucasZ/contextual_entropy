import pandas as pd
df = pd.read_csv("prob.csv")
print(df.head())
print(df["continent"].value_counts())
print(df["sector"].value_counts())
print(df["market_cap"].value_counts())
