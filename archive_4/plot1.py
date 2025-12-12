import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from utils import setupPlt, color_map

setupPlt()

with open('data/exp1.json') as f1:
    results_1 = json.load(f1)
df1 = pd.DataFrame(list(results_1.items()), columns=['Context', 'Entropy'])
# df1 = df1.sort_values('Entropy', ascending=False)

# Plot 1
plt.figure(figsize=(10, 6))
sns.barplot(data=df1, y='Context', x='Entropy', palette=color_map)
plt.title('Entropy Reduction from Context Combinations')
plt.xlabel('Conditional Entropy')
plt.axvline(x=results_1["[]"], color='r', linestyle='--', label='No context')
# plt.legend()
plt.gcf().savefig("entropy1.pdf", bbox_inches="tight")
