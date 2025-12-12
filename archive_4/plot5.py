import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from utils import setupPlt, color_map

setupPlt()

with open('data/exp5.json') as f5:
    results_5 = json.load(f5)
df5 = pd.DataFrame(list(results_5.items()), columns=['Context', 'Entropy'])


plt.figure(figsize=(10, 6))
sns.barplot(data=df5, y='Context', x='Entropy', palette=color_map)
plt.title('(Qwen) Entropy Reduction from Context Combinations')
plt.xlabel('Conditional Entropy')
plt.axvline(x=results_5["[]"], color='r', linestyle='--', label='No context')
# plt.legend()
plt.gcf().savefig("entropy5.pdf", bbox_inches="tight")
