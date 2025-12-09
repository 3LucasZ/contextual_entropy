import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap

# 1. Helper Function: Hide numbers if they are too small (< 2%)


def make_autopct(pct):
    return ('%1.1f%%' % pct) if pct > 5 else ''


# Load data
df = pd.read_csv('data/prob.csv')

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# --- Pie chart for Continent ---
cont_probs = df.groupby('continent')['probability'].sum()
# Generate dynamic viridis colors based on number of categories
cont_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cont_probs)))
axs[0].pie(cont_probs,
           labels=cont_probs.index,
           autopct=make_autopct,
           colors=cont_colors,
           startangle=90)
axs[0].set_title('Probability by Continent')

# --- Pie chart for Sector ---
sect_probs = df.groupby('sector')['probability'].sum()
sect_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sect_probs)))
axs[1].pie(sect_probs,
           labels=(sect_probs.index),
           autopct=make_autopct,
           colors=sect_colors,
           startangle=90)
axs[1].set_title('Probability by Sector')

# --- Pie chart for Market Cap ---
cap_probs = df.groupby('market_cap')['probability'].sum()
cap_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cap_probs)))
axs[2].pie(cap_probs,
           labels=(cap_probs.index),
           autopct=make_autopct,
           colors=cap_colors,
           startangle=90)
axs[2].set_title('Probability by Market Cap')

plt.tight_layout()
plt.savefig("probs.jpg")
