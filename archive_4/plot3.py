import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from utils import setupPlt, color_map
setupPlt()

with open('data/exp3.json') as f3:
    results_3 = json.load(f3)
df3 = pd.DataFrame(list(results_3.items()), columns=['Context', 'Entropy'])

# Plot 3
plt.figure(figsize=(10, 6))
sns.barplot(data=df3, y='Context', x='Entropy', palette='viridis')
plt.title('Entropy Reduction from Context Combinations + perturbed question')
plt.xlabel('Conditional Entropy')
plt.axvline(x=results_3["[]"], color='r', linestyle='--', label='No context')
plt.legend()
plt.tight_layout()
plt.savefig("entropy3.jpg")
