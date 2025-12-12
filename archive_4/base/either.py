import os
import seaborn as sns
import matplotlib.pyplot as plt
import re
import mlx.core as mx
import numpy as np
import pandas as pd
import json
import itertools
from mlx_lm import load, generate

model_id = "mlx-community/Qwen2.5-7B-Instruct-4bit"

system = "You are a professional stock analyst and trader. You must only answer with either overperform or underperform."
question = "Will this stock overperform or underperform the median stock over the next 12 months?"

token1 = "Under"
token2 = "Over"
quick = True

print(f"Loading {model_id}...")
model, tokenizer = load(model_id)
yes_id = tokenizer.encode(token1)[0]
no_id = tokenizer.encode(token2)[0]


def analyze_llm(prompt, verbose=False):
    messages = [
        {"role": "system",
            "content": system},
        {"role": "user", "content": prompt}
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = mx.array([tokenizer.encode(full_prompt)])
    logits = model(input_ids)
    # Get the logits for the last token
    last_token_logits = logits[0, -1, :]
    yes_score = last_token_logits[yes_id].item()
    no_score = last_token_logits[no_id].item()
    # neither score
    logits_np = np.array(last_token_logits)
    logits_np[yes_id] = -np.inf
    logits_np[no_id] = -np.inf
    neither_score = np.logaddexp.reduce(logits_np)
    # Softmax on the two tokens
    scores = np.array([yes_score, no_score, neither_score])
    probs = np.exp(scores) / np.sum(np.exp(scores))
    p_yes, p_no, p_neither = probs[0], probs[1], probs[2]

    # entropy
    if p_yes == 0 or p_no == 0:
        entropy = 0.0
    else:
        entropy = -(p_yes * np.log2(p_yes) + p_no *
                    np.log2(p_no) + p_neither * np.log2(p_neither))

    #  actual text response
    response_text = generate(
        model,
        tokenizer,
        prompt=full_prompt,
        max_tokens=10,  # Adjust
        verbose=False
    )

    if verbose:
        print(
            f"P({token1}): {p_yes:.2%}, P({token2}): {p_no:.2%}, P(neither): {p_neither:.2%}, H: {entropy:.4f}")
        print(f"Reply: {response_text.strip()}")

    return p_yes, p_no, entropy, response_text.strip()


prob_df = pd.read_csv("data/prob.csv")

# List to store all rows for the final CSV
detailed_logs = []


def run_trial(columns, verbose=False):
    # Group by the specified columns
    df = prob_df[columns + ["probability"]
                 ].groupby(columns).sum().reset_index()

    weighted_entropy_sum = 0
    trial_name = str(columns)

    if verbose:
        print(f"\n--- Running Trial: {trial_name} ---")

    for index, row in df.iterrows():
        # Build the prompt dynamically
        context_str = []
        if 'continent' in columns:
            context_str.append(f"is in {row['continent']}")
        if 'market_cap' in columns:
            context_str.append(f"is a {row['market_cap']} company")
        if 'sector' in columns:
            context_str.append(f"is in the {row['sector']} sector")

        # Handle the grammar for the join
        joined_context = " and ".join(context_str)
        prompt_text = f"The stock {joined_context}. {question}"

        # Run analysis
        p_yes, p_no, entropy, reply = analyze_llm(prompt_text, verbose)

        # Log data for the CSV
        detailed_logs.append({
            "Trial": trial_name,
            "Prompt": prompt_text,
            "P_Yes": p_yes,
            "P_No": p_no,
            "Entropy": entropy,
            "Reply": reply
        })

        # Calculate weighted entropy for the aggregate JSON
        prior_prob = row["probability"]
        weighted_entropy_sum += entropy * prior_prob

    return weighted_entropy_sum


if __name__ == "__main__":
    verbose = True
    out = {}

    # Run Control (Empty context)
    print(f"\n--- Running Control Group ---")
    control_prompt = f"The stock is unknown. {question}"
    p_yes, p_no, entropy, reply = analyze_llm(control_prompt, verbose)
    out["[]"] = entropy
    detailed_logs.append({
        "Trial": "[]",
        "Prompt": control_prompt,
        "P_Yes": p_yes,
        "P_No": p_no,
        "Entropy": entropy,
        "Reply": reply
    })

    # Run Combinations
    column_names = ["continent", "market_cap", "sector"]
    for r in range(1, len(column_names) + (-1 if quick else 1)):
        for subset_names in itertools.combinations(column_names, r):
            cols = list(subset_names)
            tot = run_trial(cols, verbose)
            out[str(cols)] = tot

    # Save Summary JSON
    print("Saving test.json...")
    with open("data/test.json", "w") as json_file:
        json.dump(out, json_file, indent=4)

    # Save Detailed CSV
    print("Saving test.csv...")
    results_df = pd.DataFrame(detailed_logs)
    results_df.to_csv("data/test.csv", index=False)
    print("Done.")


# Load data
df = pd.read_csv("data/test.csv")


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
    filename = "fig/test.jpg"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


generate_combined_plot(df)
