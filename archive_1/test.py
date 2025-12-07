import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen1.5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", trust_remote_code=True).eval()

yes_token_id = tokenizer.encode("Yes")[0]
no_token_id = tokenizer.encode("No")[0]


def get_llm_entropy(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits of last token
    next_token_logits = outputs.logits[0, -1, :]
    # Yes / No logits
    yes_logit = next_token_logits[yes_token_id].item()
    no_logit = next_token_logits[no_token_id].item()
    # Softmax; probabilities normalized
    probs = F.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
    p_yes = probs[0].item()
    p_no = probs[1].item()
    # Entropy
    if p_yes == 0 or p_no == 0:
        return 0.0
    h = -(p_yes * np.log2(p_yes) + p_no * np.log2(p_no))
    return h


if __name__ == "__main__":
    get_llm_entropy("Is the world flat? Answer Yes or No.")
