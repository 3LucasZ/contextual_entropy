import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import load

model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
print(f"Loading {model_id}...")
model, tokenizer = load(model_id)

# system = "You are a professional stock analyst and trader. You must only answer with either overperform or underperform."
# question = "Will this stock overperform or underperform the median stock over the next 12 months?"

system = "You must answer with only overperform or underperform."
question = "Will this stock overperform or underperform the median stock over the next 12 months?"


def analyze_llm(prompt, verbose=False):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Prepare input
    input_ids = mx.array([tokenizer.encode(full_prompt)])

    # Forward pass
    logits = model(input_ids)

    # 1. Get logits for the very last token generated
    last_token_logits = logits[0, -1, :]

    # 2. Apply Softmax to convert logits to probabilities
    probs = mx.softmax(last_token_logits)

    # 3. Sort to find the top K most likely tokens
    top_k = 5
    # argsort returns indices of sorted values; [::-1] reverses to get descending order
    sorted_indices = mx.argsort(probs)[::-1]

    print(f"\n--- Probability Distribution for '{prompt[:20]}...' ---")

    for i in range(top_k):
        token_id = sorted_indices[i].item()
        probability = probs[token_id].item()

        # Decode the token ID to text
        token_text = tokenizer.decode([token_id])

        print(
            f"Rank {i+1}: Token '{token_text}' | Probability: {probability:.2%}")


analyze_llm(f"The stock is unknown. {question}", True)
