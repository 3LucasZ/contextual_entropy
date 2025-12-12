import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils import setupPlt
setupPlt()

# Load data
df = pd.read_csv("data/exp5.csv")


def extract_info(row):
    trial = row['Trial'][2:-2]
    prompt = row['Prompt']

    # Identify Group & Strata via Regex
    if trial == "continent":
        group = "Continent"
        match = re.search(r"The stock is in (.*?)\. Will", prompt)
        strata = match.group(1) if match else "Unknown"

    elif trial == "market_cap":
        group = "Market Cap"
        match = re.search(r"The stock is a (.*?) company\. Will", prompt)
        strata = match.group(1) if match else "Unknown"

    elif trial == "sector":
        group = "Sector"
        match = re.search(r"The stock is in the (.*?) sector\. Will", prompt)
        strata = match.group(1) if match else "Unknown"
        strata = strata.replace(" ", "\n")
        if "Info" in strata:
            strata = "Info Tech"
        if "Discretionary" in strata:
            strata = "Consumer Disc"
        if "Communication" in strata:
            strata = "Comms Services"
    else:
        group = "ERROR"
        strata = "ERROR"
    return pd.Series([group, strata])


# Apply extraction logic
df[['Group', 'Strata']] = df.apply(extract_info, axis=1)
df = df[df['Group'] != "ERROR"]


def generate_combined_plot(df):
    sns.set_theme(style="whitegrid")

    # Filter out Control for grouping plots
    groups = df[df["Group"] != "Control"]["Group"].unique()

    # Create a figure with 3 subplots arranged horizontally
    # sharey=True ensures all plots share the same Y-axis scale for easy comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Use zip to iterate over axes and groups simultaneously
    # This assumes there are exactly 3 groups (Continent, Market Cap, Sector)
    for ax, group in zip(axes, groups):
        subset = df[df["Group"] == group].copy()

        # Sort by Entropy descending
        subset = subset.sort_values("Entropy", ascending=False)

        # Bar Plot on the specific axis (ax=ax)
        sns.barplot(
            data=subset,
            x="Strata",
            y="Entropy",
            hue="Strata",
            palette="viridis",
            ax=ax,
            dodge=False
        )

        # Clean up: Remove legend (since x-axis labels are sufficient)
        if ax.get_legend():
            ax.get_legend().remove()

        ax.set_title(f"Entropy Reduction Per {group}")
        ax.set_xlabel(group)

        # Set Y-label only for the first subplot to reduce clutter
        if ax == axes[0]:
            ax.set_ylabel("Entropy")
        else:
            ax.set_ylabel("")

        # Add numeric labels on top of bars
        for i, v in enumerate(subset["Entropy"]):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

        # Rotate x-axis labels for readability
        ax.tick_params(axis='x', labelrotation=30)

    plt.tight_layout()
    filename = "entropy5_2.jpg"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


if __name__ == "__main__":
    generate_combined_plot(df)
