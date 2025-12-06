import itertools
import json
import pandas as pd

from query_mlx import get_llm_entropy

prob_df = pd.read_csv("prob.csv")


def run_trial(columns, verbose=False):
    df = prob_df[columns + ["probability"]
                 ].groupby(columns).sum().reset_index()
    tot = 0
    if verbose:
        print(df.head())
    for index, row in df.iterrows():
        context_str = []
        if 'continent' in columns:
            context_str += [f"is in {row['continent']}"]
        if 'market_cap' in columns:
            context_str += [f"is a {row['market_cap']} company"]
        if 'sector' in columns:
            context_str += [f"is in the {row['sector']} sector"]
        prompt = f"The stock {" and ".join(context_str)}. Will the stock go up?"
        if verbose:
            print(prompt)
        h = get_llm_entropy(prompt, verbose)
        p = row["probability"]
        tot += h * p
    return tot


if __name__ == "__main__":
    verbose = True

    out = {}
    column_names = ["continent", "market_cap", "sector"]
    control = get_llm_entropy(
        "The stock is unknown. Will the stock go up?", verbose)
    out["[]"] = control
    print(out)
    for r in range(1, len(column_names) + 1):
        for subset_names in itertools.combinations(column_names, r):
            cols = list(subset_names)
            print(cols)
            tot = run_trial(cols, verbose)
            out[str(cols)] = tot
    with open("out.json", "w") as json_file:
        json.dump(out, json_file, indent=4)
