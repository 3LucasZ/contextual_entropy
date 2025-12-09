import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json


with open('data/exp1.json') as f1:
    results_1 = json.load(f1)
df1 = pd.DataFrame(list(results_1.items()), columns=['Context', 'Entropy'])
# df1 = df1.sort_values('Entropy', ascending=False)

with open('data/exp2.json') as f2:
    results_2 = json.load(f2)
df2 = pd.DataFrame(list(results_2.items()), columns=['Context', 'Entropy'])
# df2 = df2.sort_values('Entropy', ascending=False)

# Plot 1
plt.figure(figsize=(10, 6))
sns.barplot(data=df1, y='Context', x='Entropy', palette='viridis')
plt.title('Experiment 1', fontsize=14)
plt.xlabel('Conditional Entropy', fontsize=12)
plt.axvline(x=results_1["[]"], color='r', linestyle='--', label='Baseline')
plt.legend()
plt.tight_layout()
plt.savefig("entropy1.jpg")

# Plot 2
plt.figure(figsize=(10, 6))
sns.barplot(data=df2, y='Context', x='Entropy', palette='viridis')
plt.xlim(5)
plt.title('Experiment 2', fontsize=14)
plt.xlabel('Conditional Entropy', fontsize=12)
plt.axvline(x=results_2["[]"], color='r', linestyle='--', label='Baseline')
plt.legend()
plt.tight_layout()
plt.savefig("entropy2.jpg")
