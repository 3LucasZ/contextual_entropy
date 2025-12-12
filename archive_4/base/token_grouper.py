import mlx.core as mx
import numpy as np
import pandas as pd
import json
import itertools
from mlx_lm import load, generate

model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
print(f"Loading {model_id}...")
model, tokenizer = load(model_id)

# token IDs
yes = ["Yes"]
no = ["No"]


def get_token_ids(words):
    ids = []
    for w in words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return list(set(ids))


yes_ids = get_token_ids(yes)
no_ids = get_token_ids(no)

# system = "You are a professional stock analyst and trader. You must only answer with either overperform or underperform."
# question = "Will this stock overperform or underperform the median stock over the next 12 months?"

system = "You are a professional stock analyst and trader. Answer Yes or No."
question = "Will this stock outperform the median stock over the next 12 months?"


def analyze_llm(prompt, verbose=False):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = mx.array([tokenizer.encode(full_prompt)])
    logits = model(input_ids)
    # Get the logits for the last token
    last_token_logits = logits[0, -1, :]
    yes_logits = np.array([last_token_logits[i].item() for i in yes_ids])
    no_logits = np.array([last_token_logits[i].item() for i in no_ids])
    all_target_logits = np.concatenate([yes_logits, no_logits])
    max_logit = np.max(all_target_logits)
    sum_exp_under = np.sum(np.exp(yes_logits - max_logit))
    sum_exp_over = np.sum(np.exp(no_logits - max_logit))
    total_sum = sum_exp_under + sum_exp_over
    p_yes = sum_exp_under / total_sum
    p_no = sum_exp_over / total_sum

    # entropy
    if p_yes == 0 or p_no == 0:
        entropy = 0.0
    else:
        entropy = -(p_yes * np.log2(p_yes) + p_no * np.log2(p_no))

    #  actual text response
    response_text = generate(
        model,
        tokenizer,
        prompt=full_prompt,
        max_tokens=50,  # Adjust
        verbose=False
    )

    if verbose:
        print(f"P(Yes): {p_yes:.2%}, P(No): {p_no:.2%}, H: {entropy:.4f}")
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
    for r in range(1, len(column_names) + 1):
        for subset_names in itertools.combinations(column_names, r):
            cols = list(subset_names)
            tot = run_trial(cols, verbose)
            out[str(cols)] = tot

    # Save Summary JSON
    print("Saving exp1.json...")
    with open("data/exp1.json", "w") as json_file:
        json.dump(out, json_file, indent=4)

    # Save Detailed CSV
    print("Saving exp1.csv...")
    results_df = pd.DataFrame(detailed_logs)
    results_df.to_csv("data/exp1.csv", index=False)
    print("Done.")
