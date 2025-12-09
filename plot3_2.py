import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

# Load Data 1
with open('data/exp1.json') as f1:
    results_1 = json.load(f1)
df1 = pd.DataFrame(list(results_1.items()), columns=['Context', 'Entropy'])
df1['Experiment'] = 'prompt: median'  # Add identifier

# Load Data 3_2
with open('data/exp3_2.json') as f3_2:
    results_3_2 = json.load(f3_2)
df3_2 = pd.DataFrame(list(results_3_2.items()), columns=['Context', 'Entropy'])
df3_2['Experiment'] = 'prompt: average'  # Add identifier

# Combine the DataFrames
df_combined = pd.concat([df1, df3_2])

# Optional: Determine sort order based on df1 values for a cleaner plot
# This ensures contexts are listed in descending order of entropy from Exp 1
order = df1.sort_values('Entropy', ascending=False)['Context']

# Plot
plt.figure(figsize=(12, 7))
sns.barplot(
    data=df_combined,
    y='Context',
    x='Entropy',
    hue='Experiment',
    palette='viridis',
    order=order
)

plt.title('Entropy Reduction Comparison')
plt.xlabel('Conditional Entropy')

# Add vertical lines for "No context" baselines
# We use different line styles to distinguish the two baselines
# if "[]" in results_1:
#     plt.axvline(x=results_1["[]"], color='purple',
#                 linestyle='--', alpha=0.7, label='No context (Exp 1)')
# if "[]" in results_3_2:
#     plt.axvline(x=results_3_2["[]"], color='teal', linestyle=':',
#                 linewidth=2.5, alpha=0.9, label='No context (Exp 3_2)')

plt.legend()
plt.tight_layout()
plt.savefig("entropy3_2.jpg")
