import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Your results
results = {
    "Baseline": 0.004176,
    "Region": 0.007151,
    "Cap": 0.004680,
    "Sector": 0.014386,
    "Region + Cap": 0.009968,
    "Region + Sector": 0.040887,
    "Cap + Sector": 0.007278,
    "Full Context": 0.024815
}

df_res = pd.DataFrame(list(results.items()), columns=['Context', 'Entropy'])
df_res = df_res.sort_values('Entropy', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df_res, y='Context', x='Entropy', palette='viridis')
plt.title('The "Socratic Effect": Context Increases Model Uncertainty', fontsize=14)
plt.xlabel('Weighted Entropy (Bits)', fontsize=12)
plt.axvline(x=0.004176, color='r', linestyle='--', label='Baseline Bias')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("entropy.jpg")
