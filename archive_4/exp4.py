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
yes_id = tokenizer.encode("Yes")[0]
no_id = tokenizer.encode("No")[0]

question = "Will this stock underperform the median stock over the next 12 months?"


def analyze_llm(prompt, verbose=False):
    messages = [
        {"role": "system", "content": "You are a professional stock analyst and trader. Answer Yes or No."},
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
    # Softmax on the two tokens
    scores = np.array([yes_score, no_score])
    scores -= np.max(scores)
    probs = np.exp(scores) / np.sum(np.exp(scores))
    p_yes, p_no = probs[0], probs[1]

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
        for col in columns:
            if 'continent' == col:
                context_str.append(f"is in {row['continent']}")
            if 'market_cap' == col:
                context_str.append(f"is a {row['market_cap']} company")
            if 'sector' == col:
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

    # Run Combinations
    column_names = ["continent", "market_cap", "sector"]
    for subset_names in itertools.permutations(column_names, 3):
        cols = list(subset_names)
        tot = run_trial(cols, verbose)
        out[str(cols)] = tot

    # Save Summary JSON
    print("Saving exp4.json...")
    with open("data/exp4.json", "w") as json_file:
        json.dump(out, json_file, indent=4)

    # Save Detailed CSV
    print("Saving exp4.csv...")
    results_df = pd.DataFrame(detailed_logs)
    results_df.to_csv("data/exp4.csv", index=False)
    print("Done.")
