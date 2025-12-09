import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


financial_df = pd.read_csv('data/sector.csv')
llm_df = pd.read_csv('data/exp1.csv')
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

# Calculate Absolute Percent Change
# Using '1Y Change' because the prompt asked about the next 12 months.
# cols = ['1D Change', '1Y Change', 'Market Cap']
col = '1Y Change'

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
plt.title(f'Model Entropy vs. Absolute Magnitude of {col} for sectors')
plt.xlabel('Conditional Entropy (Uncertainty)')
plt.ylabel(f'Absolute {col} (Accuracy)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("accuracy_scatter.jpg")

correlation = merged_df['Entropy'].corr(merged_df['Abs_Change'])
print(f"Correlation between Entropy and Absolute {col}: {correlation:.4f}")
