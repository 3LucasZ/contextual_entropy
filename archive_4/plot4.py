import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from utils import setupPlt

setupPlt()

with open('data/exp4.json') as f4:
    results_4 = json.load(f4)
df4 = pd.DataFrame(list(results_4.items()), columns=['Context', 'Entropy'])
df4 = df4.sort_values('Entropy', ascending=True)


# Plot 4
plt.figure(figsize=(10, 6))
sns.barplot(data=df4, y='Context', x='Entropy',
            palette='magma_r', hue="Entropy")
plt.title('Entropy Reduction from Context Permutations')
plt.xlabel('Conditional Entropy')
plt.gca().get_legend().set_visible(False)
plt.gcf().savefig("entropy4.pdf", bbox_inches="tight")
