import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from utils import extract_info, setupPlt, color_map


class Plotter:
    def __init__(self):
        setupPlt()

    def plot(self, title, fin, fout, sort):
        with open(f'data/{fin}.json') as f:
            res = json.load(f)
        df = pd.DataFrame(list(res.items()),
                          columns=['Context', 'Entropy'])
        if sort:
            df = df.sort_values('Entropy', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, y='Context', x='Entropy', palette=color_map)
        plt.title(title)
        plt.xlabel('Conditional Entropy')
        filename = f"fig/{fout}.pdf"
        plt.gcf().savefig(filename, bbox_inches="tight")

    def plot_cat(self, title, fin, fout, sort):
        df = pd.read_csv(f"data/{fin}.csv")
        df[['Group', 'Strata']] = df.apply(extract_info, axis=1)
        df = df[df['Group'] != "ERROR"]
        groups = df[df["Group"] != "Control"]["Group"].unique()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        for ax, group in zip(axes, groups):
            subset = df[df["Group"] == group].copy()

            # Sort by Entropy descending
            if sort:
                subset = subset.sort_values("Entropy", ascending=False)

            sns.barplot(
                data=subset, x="Strata", y="Entropy", hue="Strata",
                palette="viridis", ax=ax, dodge=False
            )

            # Remove legend
            if ax.get_legend():
                ax.get_legend().remove()

            ax.set_title("Entropy Reduction Per {group}")
            ax.set_xlabel(group)

            # Y-label only for the first subplot
            if ax == axes[0]:
                ax.set_ylabel("Entropy")
            else:
                ax.set_ylabel("")

            # Add numeric labels on top of bars
            for i, v in enumerate(subset["Entropy"]):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
            ax.tick_params(axis='x', labelrotation=30)
        filename = f"fig/{fout}.pdf"
        plt.gcf().savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot("Entropy Reduction from Context Combinations",
                 "engine", "engine", False)
    plotter.plot_cat("", "engine", "engine_cat", True)
