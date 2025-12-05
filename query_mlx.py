import mlx.core as mx
import numpy as np
from mlx_lm import load

model_id = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
print(f"Loading {model_id}...")
model, tokenizer = load(model_id)

yes_id = tokenizer.encode("Yes")[0]
no_id = tokenizer.encode("No")[0]


def get_llm_entropy(prompt, verbose=False):
    messages = [
        {"role": "system", "content": "You are a professional stock analyst and trader. Answer queries with Yes or No."},
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    # convert to MLX array
    input_ids = mx.array([tokenizer.encode(prompt)])
    # get scores
    logits = model(input_ids)
    last_token_logits = logits[0, -1, :]
    yes_score = last_token_logits[yes_id].item()
    no_score = last_token_logits[no_id].item()
    # softmax probabilities
    scores = np.array([yes_score, no_score])
    scores -= np.max(scores)
    probs = np.exp(scores) / np.sum(np.exp(scores))
    p_yes, p_no = probs[0], probs[1]
    # entropy
    if p_yes == 0 or p_no == 0:
        entropy = 0.0
    else:
        entropy = -(p_yes * np.log2(p_yes) + p_no * np.log2(p_no))
    if verbose:
        print(f"P(Yes): {p_yes:.2%}, P(No): {p_no:.2%}, H: {entropy:.4f}")
    return entropy


if __name__ == "__main__":
    # Obvious answer
    print("Entropy:", get_llm_entropy("Is the earth flat?"))
    # Confusing answer
    print("Entropy:", get_llm_entropy(
        "Will a coin flip land on heads?"))
    # Investing answer
    print("Entropy:", get_llm_entropy("Will Tesla stock go up?"))
    print("Entropy:", get_llm_entropy("Will Apple stock go up?"))
    print("Entropy:", get_llm_entropy("Will Google stock go up?"))
    print("Entropy:", get_llm_entropy(
        "Will Google stock go up? Answer Yes or No."))
    print("Entropy:", get_llm_entropy(
        "Will Google stock go up? Answer No or Yes."))
