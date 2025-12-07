import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import entropy

model_id = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", trust_remote_code=True).eval()

yes_token_id = tokenizer.encode("Yes")[0]
no_token_id = tokenizer.encode("No")[0]


def get_llm_entropy(region, ):
    results = []
    base_q = "Will the stock go up in price? Answer Yes or No."
    for index, row in sampled_df.iterrows():
        symbol = row['symbol']
        country = row['country']
        industry = row['industry']
        cap = row['market_cap']

    # 1. Baseline (No Context)
    prompt_baseline = base_q.format(symbol=symbol)
    h_base, p_base = get_llm_entropy(prompt_baseline)

    # 2. Context C1 (Country)
    prompt_c1 = f"The stock {symbol} is from {country}. " + \
        base_q.format(symbol=symbol)
    h_c1, _ = get_llm_entropy(prompt_c1)

    # 3. Context C2 (Industry)
    prompt_c2 = f"The stock {symbol} is in the {industry} industry. " + \
        base_q.format(symbol=symbol)
    h_c2, _ = get_llm_entropy(prompt_c2)

    # 4. Context Permutation (C1, C2)
    prompt_perm1 = f"Country: {country}. Industry: {industry}. " + \
        base_q.format(symbol=symbol)
    h_perm1, _ = get_llm_entropy(prompt_perm1)

    # 5. Context Permutation (C2, C1) - Lost in the Middle check
    prompt_perm2 = f"Industry: {industry}. Country: {country}. " + \
        base_q.format(symbol=symbol)
    h_perm2, _ = get_llm_entropy(prompt_perm2)

    # 6. Obvious Context (Control)
    prompt_obvious = f"Analyst rating: Strong Buy. " + \
        base_q.format(symbol=symbol)
    h_obvious, _ = get_llm_entropy(prompt_obvious)

    results.append({
        'symbol': symbol,
        'country': country,
        'industry': industry,
        'H_baseline': h_base,
        'H_country': h_c1,
        'H_industry': h_c2,
        'H_combined_1': h_perm1,
        'H_combined_2': h_perm2,
        'H_obvious': h_obvious
    })

    results_df = pd.DataFrame(results)
    # Save results
    results_df.to_csv('experiment_results.csv')


def get_llm_entropy(region, ):
    results = []
    base_q = "Will the stock go up in price? Answer Yes or No."
    for index, row in sampled_df.iterrows():
        symbol = row['symbol']
        country = row['country']
        industry = row['industry']
        cap = row['market_cap']

    # 1. Baseline (No Context)
    prompt_baseline = base_q.format(symbol=symbol)
    h_base, p_base = get_llm_entropy(prompt_baseline)

    # 2. Context C1 (Country)
    prompt_c1 = f"The stock {symbol} is from {country}. " + \
        base_q.format(symbol=symbol)
    h_c1, _ = get_llm_entropy(prompt_c1)

    # 3. Context C2 (Industry)
    prompt_c2 = f"The stock {symbol} is in the {industry} industry. " + \
        base_q.format(symbol=symbol)
    h_c2, _ = get_llm_entropy(prompt_c2)

    # 4. Context Permutation (C1, C2)
    prompt_perm1 = f"Country: {country}. Industry: {industry}. " + \
        base_q.format(symbol=symbol)
    h_perm1, _ = get_llm_entropy(prompt_perm1)

    # 5. Context Permutation (C2, C1) - Lost in the Middle check
    prompt_perm2 = f"Industry: {industry}. Country: {country}. " + \
        base_q.format(symbol=symbol)
    h_perm2, _ = get_llm_entropy(prompt_perm2)

    # 6. Obvious Context (Control)
    prompt_obvious = f"Analyst rating: Strong Buy. " + \
        base_q.format(symbol=symbol)
    h_obvious, _ = get_llm_entropy(prompt_obvious)

    results.append({
        'symbol': symbol,
        'country': country,
        'industry': industry,
        'H_baseline': h_base,
        'H_country': h_c1,
        'H_industry': h_c2,
        'H_combined_1': h_perm1,
        'H_combined_2': h_perm2,
        'H_obvious': h_obvious
    })

    results_df = pd.DataFrame(results)
    # Save results
    results_df.to_csv('experiment_results.csv')


def _get_llm_entropy(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits of the last token
    next_token_logits = outputs.logits[0, -1, :]
    # Get only Yes and No logits
    yes_logit = next_token_logits[yes_token_id].item()
    no_logit = next_token_logits[no_token_id].item()
    # Softmax to get probabilities normalized
    # P(Yes) = exp(Yes) / (exp(Yes) + exp(No))
    probs = F.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
    p_yes = probs[0].item()
    p_no = probs[1].item()
    # Calculate Binary Entropy
    if p_yes == 0 or p_no == 0:
        return 0.0
    h = -(p_yes * np.log2(p_yes) + p_no * np.log2(p_no))
    return h
