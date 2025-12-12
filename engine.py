import mlx.core as mx
import numpy as np
import pandas as pd
import json
import itertools
from mlx_lm import load

import seaborn as sns
import pandas as pd
import json


config = {
    "model_id": "llama",
    "system": "You are a professional stock analyst and trader. You must only answer with either overperform or underperform.",
    "question": "Will this stock overperform or underperform the median stock over the next 12 months?",
    "token1": "Under",
    "token2": "Over",
    "neither": True,
    "file": "engine",
    "verbose": True,
    "quick": True,
}


class Engine:
    def __init__(self, config):
        self.config = config
        llama = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
        qwen = "mlx-community/Qwen2.5-7B-Instruct-4bit"
        self.model_id = llama if config["model"] == "llama" else qwen
        self.system = config["system"]
        self.question = config["question"]
        self.token1 = config["token1"]
        self.token2 = config["token2"]
        self.neither = config["neither"]
        self.file = config["file"]
        self.verbose = config["verbose"]
        self.quick = config["quick"]

        print(f"Loading {self.model_id}...")
        self.model, self.tokenizer = load(self.model_id)
        self.yes_id = self.tokenizer.encode(self.token1)[0]
        self.no_id = self.tokenizer.encode(self.token2)[0]

        self.prob_df = pd.read_csv("data/prob.csv")
        self.column_names = ["continent", "market_cap", "sector"]
        # store all rows for the final CSV
        self.detailed_logs = []

    def analyze_llm(self, prompt):
        messages = [
            {"role": "system",
                "content": self.system},
            {"role": "user", "content": prompt}
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = mx.array([self.tokenizer.encode(full_prompt)])
        logits = self.model(input_ids)
        # Get the logits for the last token
        last_token_logits = logits[0, -1, :]
        yes_score = last_token_logits[self.yes_id].item()
        no_score = last_token_logits[self.no_id].item()
        # neither score
        logits_np = np.array(last_token_logits)
        logits_np[self.yes_id] = -np.inf
        logits_np[self.no_id] = -np.inf
        neither_score = np.logaddexp.reduce(logits_np)
        # softmax on the two tokens + neither
        scores = np.array(
            [yes_score, no_score, (neither_score if self.neither else -np.inf)])
        probs = np.exp(scores) / np.sum(np.exp(scores))
        p_yes, p_no, p_neither = probs[0], probs[1], probs[2]
        # entropy
        entropy = -(p_yes * np.log2(p_yes) + p_no *
                    np.log2(p_no) + (p_neither * np.log2(p_neither) if self.neither else 0))
        # sanity gen text response
        # response_text = generate(
        #     self.model,
        #     self.tokenizer,
        #     prompt=full_prompt,
        #     max_tokens=10,  # Adjust
        #     verbose=False
        # )
        if self.verbose:
            print(
                f"P({self.token1}): {p_yes:.2%}, P({self.token2}): {p_no:.2%}, P(neither): {p_neither:.2%}, H: {entropy:.4f}")
            # print(f"Reply: {self.response_text.strip()}")

        return p_yes, p_no, entropy, ""

    def run_control(self):
        print(f"\n--- Running Control Group ---")
        control_prompt = f"The stock is unknown. {self.question}"
        p_yes, p_no, entropy, reply = self.analyze_llm(control_prompt)
        self.detailed_logs.append({
            "Trial": "[]",
            "Prompt": control_prompt,
            "P_Yes": p_yes,
            "P_No": p_no,
            "Entropy": entropy,
            "Reply": reply
        })

    def run_trial(self, columns):
        # Group by the specified columns
        df = self.prob_df[columns + ["probability"]
                          ].groupby(columns).sum().reset_index()
        trial_name = str(columns)

        if self.verbose:
            print(f"\n--- Running Trial: {trial_name} ---")

        for index, row in df.iterrows():
            # Build the prompt dynamically
            context_str = []
            if 'continent' in columns:
                context_str.append(f"is in {row['continent']}")
            if 'market_cap' in columns:
                context_str.append(f"is a {row['market_cap']} company")
            if 'sector' in columns:
                context_str.append(f"is in the {row['sector']} sector")

            # Handle the grammar for the join
            joined_context = " and ".join(context_str)
            prompt_text = f"The stock {joined_context}. {self.question}"

            # Run analysis
            p_yes, p_no, entropy, reply = self.analyze_llm(prompt_text)

            # Log data for the CSV
            self.detailed_logs.append({
                "Trial": trial_name,
                "Prompt": prompt_text,
                "P_Yes": p_yes,
                "P_No": p_no,
                "Entropy": entropy,
                "Reply": reply
            })

    def run_combinations(self):
        self.run_control()
        for r in range(1, len(self.column_names) + (-1 if self.quick else 1)):
            for subset_names in itertools.combinations(self.column_names, r):
                cols = list(subset_names)
                self.run_trial(cols)
        self.save()

    def run_permutations(self):
        pass

    def save(self):
        print(f"Saving {self.file}.csv...")
        results_df = pd.DataFrame(self.detailed_logs)
        results_df.to_csv(f"data/{self.file}.csv", index=False)
        print("Done.")


if __name__ == "__main__":
    engine = Engine(config)
    engine.run_combinations()
    engine.save()
