import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


df = pd.read_csv('prob.csv')
model_id = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", trust_remote_code=True).eval()
yes_id = tokenizer.encode("Yes")[0]
no_id = tokenizer.encode("No")[0]


def calculate_entropy_for_prompt(prompt):
    """Returns scalar entropy value in bits"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    relevant_logits = torch.tensor([logits[yes_id], logits[no_id]])
    probs = F.softmax(relevant_logits, dim=0)

    p_yes, p_no = probs[0].item(), probs[1].item()

    if p_yes == 0 or p_no == 0:
        return 0.0
    return -(p_yes * np.log2(p_yes) + p_no * np.log2(p_no))

# --- 3. The Experiment Loop ---


def run_experiment(context_columns):
    """
    context_columns: list of columns to condition on. 
    e.g., ['region'] or ['region', 'industry', 'market_cap']
    """

    # Group data by the specific context combination (strata)
    # This gives us the unique combinations existing in the world
    groups = df.groupby(context_columns)

    weighted_entropy_sum = 0

    print(f"Running experiment for contexts: {context_columns}")

    for name, group in groups:
        # A. Calculate P(c1, c2...) - The Weight
        # The probability of this specific context appearing in the dataset
        count_in_group = len(group)
        weight = count_in_group / N

        # B. Estimate H(A | c1, c2...) - The Entropy
        # We sample k stocks from this group to estimate the group's entropy
        # If group is small, take all; otherwise take max 5 samples
        sample_size = min(len(group), 5)
        sample = group.sample(n=sample_size)

        group_entropies = []

        for _, row in sample.iterrows():
            symbol = row['symbol']

            # Construct Prompt dynamically based on context_columns
            context_str = ""
            if 'region' in context_columns:
                context_str += f"in {row['region']} "
            if 'market_cap' in context_columns:
                context_str += f"is a {row['market_cap']} cap company "
            if 'industry' in context_columns:
                context_str += f"in the {row['industry']} industry"

            prompt = f"The stock {symbol} {context_str}. Will it go up? Answer Yes or No."

            # Get Entropy
            h = calculate_entropy_for_prompt(prompt)
            group_entropies.append(h)

        # Average entropy for this specific stratum
        avg_stratum_entropy = np.mean(group_entropies)

        # C. Add to global sum: Weight * Entropy
        weighted_entropy_sum += weight * avg_stratum_entropy

    return weighted_entropy_sum

# --- 4. Compare Results ---


# Experiment A: No Context (H(A))
# We treat the whole dataset as one group
h_baseline = run_experiment([])

# Experiment B: Conditioned on Region (H(A | Region))
h_region = run_experiment(['region'])

# Experiment C: Conditioned on Region + Industry + Cap (H(A | R, I, C))
h_full_context = run_experiment(['region', 'industry', 'market_cap'])

print(f"Baseline Entropy: {h_baseline}")
print(f"Entropy given Region: {h_region}")
print(f"Entropy given Full Context: {h_full_context}")

# Information Gain (Mutual Information)
print(f"Information Gain from Region: {h_baseline - h_region}")
