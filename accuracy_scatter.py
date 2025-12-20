import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import setupPlt

setupPlt()
financial_df = pd.read_csv('data/sector.csv')
llm_df = pd.read_csv('data/yes_no.csv')
llm_df = llm_df[llm_df["Trial"] == "['sector']"]

# ---------------------------------------------------------
# 2. Data Processing
# ---------------------------------------------------------

# We need to link the two datasets.
# extract the 'Sector Name' from the LLM 'Prompt' column.
known_sectors = financial_df['Sector Name'].unique()


def get_sector(prompt):
    for sector in known_sectors:
        if sector in prompt:
            return sector
    return None


llm_df['Sector Name'] = llm_df['Prompt'].apply(get_sector)

# Merge the datasets on 'Sector Name'
merged_df = pd.merge(llm_df, financial_df, on='Sector Name')

probs = pd.read_csv("data/prob.csv")
probs = probs[["sector", "probability"]].groupby("sector").agg(sum)
merged_df = pd.merge(
    merged_df, probs, left_on="Sector Name", right_on="sector")
print(merged_df.head())

# Calculate Absolute Percent Change
# Using '1Y Change' because the prompt asked about the next 12 months.
# cols = ['1D Change', '1Y Change', 'Market Cap']
# col = 'Market Cap'
col = "1Y Change"

merged_df['Abs_Change'] = merged_df[f'{col}'].abs()

# Scatter
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=merged_df,
    x='Entropy',
    y='Abs_Change',
    hue='Sector Name',
    s=150,  # Size of the dots
    alpha=0.8
)

# Labeling
plt.title(f'Model Entropy vs. Magnitude of {col} for sectors')
plt.xlabel('Conditional Entropy (Uncertainty)')
plt.ylabel(f'Absolute {col} (Accuracy)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.gcf().savefig("fig/accuracy_scatter.pdf", bbox_inches="tight")

correlation = merged_df['Entropy'].corr(merged_df['Abs_Change'])
print(f"Correlation between Entropy and Absolute {col}: {correlation:.4f}")


# def weighted_correlation(x, y, w):
#     """
#     Calculate the weighted Pearson correlation coefficient.
#     """
#     # Calculate weighted means
#     mean_x = np.average(x, weights=w)
#     mean_y = np.average(y, weights=w)

#     # Calculate weighted covariance and variances
#     cov = np.sum(w * (x - mean_x) * (y - mean_y))
#     var_x = np.sum(w * (x - mean_x)**2)
#     var_y = np.sum(w * (y - mean_y)**2)

#     return cov / np.sqrt(var_x * var_y)


# # Calculate the weighted correlation
# w_corr = weighted_correlation(
#     merged_df['Entropy'],
#     merged_df['Abs_Change'],
#     merged_df['probability']
# )

# print(f"Weighted Correlation between Entropy and Absolute {col}: {w_corr:.4f}")
