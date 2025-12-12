import mlx.core as mx
import numpy as np
import pandas as pd
import json
import itertools
from mlx_lm import load, generate

# Load model
model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
print(f"Loading {model_id}...")
model, tokenizer = load(model_id)

# map the string representation of numbers 0-100 to their token IDs
numeric_tokens = {}
valid_indices = []

question = "What is the probability (%) that this stock underperforms the median stock over the next 12 months?"

print("Mapping numeric tokens (0-100)...")
for i in range(101):
    s = str(i)
    tokens = tokenizer.encode(s)
    tid = tokens[-1]
    numeric_tokens[i] = tid
    valid_indices.append(tid)
valid_indices = mx.array(valid_indices)


def analyze_llm_numeric(prompt, verbose=False):
    messages = [
        {"role": "system",
            "content": "You are a professional stock analyst. Answer with only an integer between 0 and 100 (inclusive)."},
        {"role": "user", "content": prompt}
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Logits
    input_ids = mx.array([tokenizer.encode(full_prompt)])
    logits = model(input_ids)

    # last generated token
    last_token_logits = logits[0, -1, :]

    # ONLY for the numbers 0-100
    target_logits = last_token_logits[valid_indices]
    target_logits_np = np.array(target_logits.tolist())
    # Softmax over the domain [0, 100]
    target_logits_np -= np.max(target_logits_np)
    probs = np.exp(target_logits_np) / np.sum(np.exp(target_logits_np))

    # Expected Value
    values = np.arange(101)
    expected_value = np.sum(values * probs)

    # Entropy
    nonzero_probs = probs[probs > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))

    # Generate Complete Reply
    response_text = generate(
        model,
        tokenizer,
        prompt=full_prompt,
        max_tokens=10,
        verbose=False
    )

    if verbose:
        print(f"Exp Value: {expected_value:.2f}, H: {entropy:.4f}")
        print(f"Reply: {response_text.strip()}")

    return expected_value, entropy, response_text.strip()


prob_df = pd.read_csv("data/prob.csv")
detailed_logs = []


def run_trial(columns, verbose=False):
    df = prob_df[columns + ["probability"]
                 ].groupby(columns).sum().reset_index()

    weighted_entropy_sum = 0
    trial_name = str(columns)

    if verbose:
        print(f"\n--- Running Trial: {trial_name} ---")

    for index, row in df.iterrows():
        context_str = []
        if 'continent' in columns:
            context_str.append(f"is in {row['continent']}")
        if 'market_cap' in columns:
            context_str.append(f"is a {row['market_cap']} company")
        if 'sector' in columns:
            context_str.append(f"is in the {row['sector']} sector")

        joined_context = " and ".join(context_str)
        prompt_text = f"The stock {joined_context}. {question}"
        if verbose:
            print(prompt_text)
        exp_val, entropy, reply = analyze_llm_numeric(prompt_text, verbose)

        detailed_logs.append({
            "Trial": trial_name,
            "Prompt": prompt_text,
            "Expected_Score": exp_val,
            "Entropy": entropy,
            "Reply": reply
        })

        prior_prob = row["probability"]
        weighted_entropy_sum += entropy * prior_prob

    return weighted_entropy_sum


if __name__ == "__main__":
    verbose = True
    out = {}

    # Control
    print(f"\n--- Running Control Group ---")
    control_prompt = f"The stock is unknown. {question}"
    if verbose:
        print(control_prompt)
    exp_val, entropy, reply = analyze_llm_numeric(control_prompt, verbose)

    out["[]"] = entropy
    detailed_logs.append({
        "Trial": "[]",
        "Prompt": control_prompt,
        "Expected_Score": exp_val,
        "Entropy": entropy,
        "Reply": reply
    })

    # Combinations
    column_names = ["continent", "market_cap", "sector"]

    for r in range(1, len(column_names) + 1):
        for subset_names in itertools.combinations(column_names, r):
            cols = list(subset_names)
            tot = run_trial(cols, verbose)
            out[str(cols)] = tot

    # Save
    print("\nSaving exp2.json...")
    with open("data/exp2.json", "w") as json_file:
        json.dump(out, json_file, indent=4)

    print("Saving exp2.csv...")
    results_df = pd.DataFrame(detailed_logs)
    results_df.to_csv("data/exp2.csv", index=False)
    print("Done.")
