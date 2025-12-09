import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

with open('data/exp4.json') as f4:
    results_4 = json.load(f4)
df4 = pd.DataFrame(list(results_4.items()), columns=['Context', 'Entropy'])


# Plot 4
plt.figure(figsize=(10, 6))
sns.barplot(data=df4, y='Context', x='Entropy', palette='viridis')
plt.title('Experiment 4', fontsize=14)
plt.xlabel('Conditional Entropy', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("entropy4.jpg")
