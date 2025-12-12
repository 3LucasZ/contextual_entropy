import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from utils import setupPlt, color_map

setupPlt()


with open('data/exp2.json') as f2:
    results_2 = json.load(f2)
df2 = pd.DataFrame(list(results_2.items()), columns=['Context', 'Entropy'])
# df2 = df2.sort_values('Entropy', ascending=False)


# Plot 2
plt.figure(figsize=(10, 6))
sns.barplot(data=df2, y='Context', x='Entropy', palette='viridis')
plt.xlim(5)
plt.title('Experiment 2')
plt.xlabel('Conditional Entropy')
plt.axvline(x=results_2["[]"], color='r', linestyle='--', label='Baseline')
plt.legend()
plt.tight_layout()
plt.savefig("entropy2.jpg")
