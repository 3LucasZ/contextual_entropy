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

with open('data/exp3.json') as f3:
    results_3 = json.load(f3)
df3 = pd.DataFrame(list(results_3.items()), columns=['Context', 'Entropy'])

with open('data/exp3_2.json') as f3_2:
    results_3_2 = json.load(f3_2)
df3_2 = pd.DataFrame(list(results_3_2.items()), columns=['Context', 'Entropy'])

with open('data/exp4.json') as f4:
    results_4 = json.load(f4)
df4 = pd.DataFrame(list(results_4.items()), columns=['Context', 'Entropy'])

# Plot 1
plt.figure(figsize=(10, 6))
sns.barplot(data=df1, y='Context', x='Entropy', palette='viridis')
plt.title('Entropy Reduction from Context Combinations')
plt.xlabel('Conditional Entropy')
plt.axvline(x=results_1["[]"], color='r', linestyle='--', label='No context')
plt.legend()
plt.tight_layout()
plt.savefig("entropy1.jpg")

# Plot 1.2 in another file

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

# Plot 3
plt.figure(figsize=(10, 6))
sns.barplot(data=df3, y='Context', x='Entropy', palette='viridis')
plt.title('Entropy Reduction from Context Combinations + perturbed question')
plt.xlabel('Conditional Entropy')
plt.axvline(x=results_3["[]"], color='r', linestyle='--', label='No context')
plt.legend()
plt.tight_layout()
plt.savefig("entropy3.jpg")

# Plot 3_2
plt.figure(figsize=(10, 6))
sns.barplot(data=df3_2, y='Context', x='Entropy', palette='viridis')
plt.title('Entropy Reduction from Context Combinations + perturbed question')
plt.xlabel('Conditional Entropy')
plt.axvline(x=results_3_2["[]"], color='r', linestyle='--', label='No context')
plt.legend()
plt.tight_layout()
plt.savefig("entropy3_2.jpg")
