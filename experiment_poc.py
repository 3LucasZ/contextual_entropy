import pandas as pd

from query_mlx import get_llm_entropy

prob_df = pd.read_csv("prob.csv")


def run_experiment_single(column, verbose=False):
    df = prob_df[[column, "probability"]].groupby(column).sum()
    tot = 0
    for index, row in df.iterrows():
        prompt = f"Company data: {column} is {index}. Will the stock go down?"
        if verbose:
            print(prompt)
        h = get_llm_entropy(prompt, verbose)
        p = row["probability"]
        tot += h * p
    return tot


if __name__ == "__main__":
    verbose = True
    tot = run_experiment_single("continent", verbose)
    print(tot)
    print()
    tot = run_experiment_single("sector", verbose)
    print(tot)
    print()
    tot = run_experiment_single("market_cap", verbose)
    print(tot)


# total_weighted_entropy = 0

# for index, row in df.iterrows():
#     # 1. Get the weight P(c)
#     weight = row['probability']

#     # 2. Get the Context
#     country = row['country']
#     industry = row['industry']
#     cap = row['market_cap']

#     # 3. Measure Entropy H(A|c) using the LLM
#     # (You run the LLM on a generic prompt with this specific context)
#     prompt = f"Context: {cap} company in Industry: {industry} from {country}. Should I invest in the stock?"
#     h_context = get_llm_entropy(prompt)

#     # 4. Add to Sum
#     total_weighted_entropy += weight * h_context

# print(f"Final System Entropy: {total_weighted_entropy}")


# def run_experiment(df, context_columns):
#     # Group data by the specific context combination (strata)
#     # This gives us the unique combinations existing in the world
#     groups = df.groupby(context_columns)

#     weighted_entropy_sum = 0

#     print(f"Running experiment for contexts: {context_columns}")

#     for name, group in groups:
#         # A. Calculate P(c1, c2...) - The Weight
#         # The probability of this specific context appearing in the dataset
#         count_in_group = len(group)
#         weight = count_in_group / N

#         # B. Estimate H(A | c1, c2...) - The Entropy
#         # We sample k stocks from this group to estimate the group's entropy
#         # If group is small, take all; otherwise take max 5 samples
#         sample_size = min(len(group), 5)
#         sample = group.sample(n=sample_size)

#         group_entropies = []

#         for _, row in sample.iterrows():
#             symbol = row['symbol']

#             # Construct Prompt dynamically based on context_columns
#             context_str = ""
#             if 'region' in context_columns:
#                 context_str += f"in {row['region']} "
#             if 'market_cap' in context_columns:
#                 context_str += f"is a {row['market_cap']} cap company "
#             if 'industry' in context_columns:
#                 context_str += f"in the {row['industry']} industry"

#             prompt = f"The stock {symbol} {context_str}. Will it go up? Answer Yes or No."

#             # Get Entropy
#             h = calculate_entropy_for_prompt(prompt)
#             group_entropies.append(h)

#         # Average entropy for this specific stratum
#         avg_stratum_entropy = np.mean(group_entropies)

#         # C. Add to global sum: Weight * Entropy
#         weighted_entropy_sum += weight * avg_stratum_entropy

#     return weighted_entropy_sum
