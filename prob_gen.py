import pandas as pd
import country_converter as coco
import numpy as np

# Load data
df = pd.read_csv('data/equities.csv')

# Remove companies missing values in relevant columns
relevant_columns = ['country', 'sector', 'market_cap']
df = df.dropna(subset=relevant_columns).copy()

# Map country to continent
cc = coco.CountryConverter()
df['continent'] = cc.convert(
    names=df['country'].tolist(), to='continent_7', not_found="na")
df = df[df['continent'] != 'na']

# Configuration
n_iterations = 1000
alpha = 0.05  # 95% confidence interval

original_counts = df.groupby(['continent', 'sector', 'market_cap']).size()
full_index = original_counts.index


bootstrap_results = []
for i in range(n_iterations):
    # Sample with replacement
    sample = df.sample(frac=1, replace=True)
    # Calculate proportions
    counts = sample.groupby(['continent', 'sector', 'market_cap']).size()
    # Reindex; if a sector is missing in the sample, it gets 0 instead of being dropped
    proportions = counts.reindex(full_index, fill_value=0) / len(sample)
    bootstrap_results.append(proportions)

# DataFrame where each column is one bootstrap iteration
bootstrap_df = pd.concat(bootstrap_results, axis=1)

# Store point estimate, lower bound, upper bound
results = pd.DataFrame(index=full_index)
results['probability'] = original_counts / len(df)
results['lower'] = bootstrap_df.quantile(alpha / 2, axis=1)
results['upper'] = bootstrap_df.quantile(1 - (alpha / 2), axis=1)

# Save
results.to_csv("prob.csv")
print("Successfully saved to 'prob.csv'")
print(results.head())
