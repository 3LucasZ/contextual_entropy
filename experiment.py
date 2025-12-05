import itertools
import pandas as pd

from query_mlx import get_llm_entropy

prob_df = pd.read_csv("prob.csv")


def run_trial(columns, verbose=False):
    df = prob_df[columns + ["probability"]
                 ].groupby(columns).sum().reset_index()
    tot = 0
    print(df.head())
    for index, row in df.iterrows():
        context_str = []
        if 'continent' in columns:
            context_str += [f"is in {row['continent']}"]
        if 'market_cap' in columns:
            context_str += [f"is a {row['market_cap']} company"]
        if 'sector' in columns:
            context_str += [f"is in the {row['sector']} sector"]
        prompt = f"The stock {" and ".join(context_str)}. Will the stock go down?"
        if verbose:
            print(prompt)
        h = get_llm_entropy(prompt, verbose)
        p = row["probability"]
        tot += h * p
    return tot


if __name__ == "__main__":
    verbose = False

    results = {}
    column_names = ["continent", "market_cap", "sector"]
    for r in range(1, len(column_names) + 1):
        for subset_names in itertools.combinations(column_names, r):
            cols = list(subset_names)
            print(cols)
            tot = run_trial(cols)
            results[str(cols)] = tot
    print(results)
