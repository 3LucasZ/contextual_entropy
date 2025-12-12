import ast
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

import numpy as np

from utils import extract_info, setupPlt, color_map


class Plotter:
    def __init__(self):
        setupPlt()
        self.prob_df = pd.read_csv("data/prob.csv")

    def plot(self, title, fin, fout, sort):
        df = pd.read_csv(f"data/{fin}.csv")
        results = []

        for trial in df['Trial'].unique():
            if trial == "[]":
                continue
            columns = ast.literal_eval(trial)
            prob_df = self.prob_df[columns + ["probability", "lower", "upper"]
                                   ].groupby(columns).sum().reset_index()
            subset_df = df[df['Trial'] == trial].reset_index()
            res = prob_df.join(subset_df)
            print(res.head())
            mean = np.sum(res["probability"] * res["Entropy"])
            lower = np.sum(res["lower"] * res["Entropy"])
            upper = np.sum(res["upper"] * res["Entropy"])
            results.append({
                "Context": trial,
                "Entropy": mean,
                "Lower": lower,
                "Upper": upper,
                "Error_Min": mean - lower,
                "Error_Max": upper - mean
            })
        res_df = pd.DataFrame(results)
        if sort:
            res_df = res_df.sort_values(
                "Entropy", ascending=False)
        plt.figure(figsize=(10, 6))
        bar_colors = [color_map.get(x) for x in res_df['Context']]
        bars = plt.bar(res_df['Context'], res_df['Entropy'],
                       yerr=[res_df['Error_Min'], res_df['Error_Max']],
                       capsize=5, color=bar_colors)
        # clean up labels
        labels = [l.get_text().replace("[", "").replace("]", "").replace("'", "")
                  for l in plt.gca().get_xticklabels()]
        plt.gca().set_xticklabels(labels)
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

            ax.set_title(f"Entropy Reduction Per {group}")
            ax.set_xlabel(group)

            # Y-label only for the first subplot
            if ax == axes[0]:
                ax.set_ylabel("Entropy")
            else:
                ax.set_ylabel("")

            # Add numeric labels on top of bars
            # for i, v in enumerate(subset["Entropy"]):
            #     ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
            ax.tick_params(axis='x', labelrotation=30)
        fig.suptitle(title)
        filename = f"fig/{fout}.pdf"
        plt.gcf().savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot("Entropy Reduction from Context Combinations",
                 "yes_no", "yes_no", False)
    # plotter.plot_cat("", "engine", "engine_cat", True)
