import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(42)
N_SAMPLES = 2000


def mock_data(n):
    """
    synthetic stock data
    - Low PE + High EPS Growth => outperform
    - High Debt => underperform
    """
    # PE Ratio (normal)
    pe_ratio = np.random.normal(15, 5, n)
    pe_ratio = np.maximum(pe_ratio, 1)
    # EPS Growth (normal)
    eps_growth = np.random.normal(5, 10, n)
    # Debt (uniform)
    debt_equity = np.random.uniform(0, 2, n)
    # Noise (normal)
    noise = np.random.normal(0, 3, n)

    signal = -0.4 * (pe_ratio - 15) + 0.5 * \
        (eps_growth - 5) - 0.3 * (debt_equity - 1)
    final_score = signal + noise
    target = (final_score > 0).astype(int)
    df = pd.DataFrame({
        'PE_Ratio': pe_ratio,
        'EPS_Growth': eps_growth,
        'Debt_Equity': debt_equity,
        'Signal_Strength': signal,
        'Target': target
    })
    return final_score, df


def get_entropy(probs):
    return entropy(probs, base=2)


def get_difficulty(df, context_cols):
    """
    difficulty of a question = conditional empirical distribution
    1. Bin the continuous features
    2. Look at historical Target mean.
    3. If mean is 0.5, empirical entropy is High (Max Uncertainty).
    4. If mean is 0.9 or 0.1, empirical entropy is Low (Easy question).
    """
    df_calc = df.copy()
    # Discretize
    enc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    bin_cols = [f"{c}_bin" for c in context_cols]
    df_calc[bin_cols] = enc.fit_transform(df_calc[context_cols])
    # empirical probability
    grouped = df_calc.groupby(bin_cols)['Target'].agg(
        ['mean', 'count']).reset_index()
    # avoid statistical noise
    grouped = grouped[grouped['count'] > 10]
    # difficulty
    probs = grouped['mean'].clip(0.001, 0.999)
    grouped['Empirical_Entropy'] = - \
        (probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
    # Merge
    df_calc = df_calc.merge(grouped, on=bin_cols, how='left')
    return df_calc['Empirical_Entropy'], df_calc['mean']


def run_experiment():
    _, df = mock_data(N_SAMPLES)

    # Split train/test
    train_df = df.iloc[:1500]
    test_df = df.iloc[1500:].copy()

    # CONDITION A: LOW CONTEXT (Only PE Ratio provided)
    print("\nRunning Condition A: Low Context (PE Ratio only)...")
    cols_A = ['PE_Ratio']

    # FAKE MODEL!!
    model_A = GradientBoostingClassifier(n_estimators=100)
    model_A.fit(train_df[cols_A], train_df['Target'])
    entropy_A, empirical_prob_A = get_difficulty(
        df, cols_A)
    test_df['Empirical_Entropy_A'] = entropy_A.iloc[1500:].values

    # Predictions
    probs_A = model_A.predict_proba(test_df[cols_A])
    test_df['Model_Confidence_A'] = np.max(
        probs_A, axis=1)
    test_df['Model_Entropy_A'] = [
        get_entropy(p) for p in probs_A]

    # CONDITION B: HIGH CONTEXT (PE + EPS + Debt provided)
    print("Running Condition B: High Context (PE + EPS + Debt)...")
    cols_B = ['PE_Ratio', 'EPS_Growth', 'Debt_Equity']

    # FAKE MODEL!!
    model_B = GradientBoostingClassifier(n_estimators=100)
    model_B.fit(train_df[cols_B], train_df['Target'])

    entropy_B, empirical_prob_B = get_difficulty(
        df, cols_B)
    test_df['Empirical_Entropy_B'] = entropy_B.iloc[1500:].values

    # Predictions
    probs_B = model_B.predict_proba(test_df[cols_B])
    test_df['Model_Confidence_B'] = np.max(probs_B, axis=1)
    test_df['Model_Entropy_B'] = [
        get_entropy(p) for p in probs_B]

    return test_df


def visualize_results(df):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    avg_entropy = df[['Model_Entropy_A', 'Model_Entropy_B']].mean()
    sns.barplot(x=['Low Context\n(PE Only)', 'High Context\n(Full Financials)'],
                y=avg_entropy.values, palette='viridis')
    plt.title("Hypothesis Test: Does Context Reduce Model Entropy?")
    plt.ylabel("Avg Model Entropy (Uncertainty)")

    plt.subplot(2, 2, 2)
    sns.regplot(x='Empirical_Entropy_B', y='Model_Confidence_B', data=df,
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title("Validation: Model Confidence vs. Question Difficulty")
    plt.xlabel("Empirical Difficulty (Historical Entropy)")
    plt.ylabel("Model Confidence (Max Prob)")

    plt.subplot(2, 2, 3)
    sns.kdeplot(df['Model_Confidence_A'], label='Low Context', fill=True)
    sns.kdeplot(df['Model_Confidence_B'], label='High Context', fill=True)
    plt.title("Distribution of Model Confidence Scores")
    plt.xlabel("Confidence (0.5 = Clueless, 1.0 = Certain)")
    plt.legend()

    plt.subplot(2, 2, 4)
    df['Entropy_Reduction'] = df['Model_Entropy_A'] - df['Model_Entropy_B']
    sns.scatterplot(x='Empirical_Entropy_A', y='Entropy_Reduction',
                    data=df, alpha=0.5, hue='Target')
    plt.title("Where does Context help most?")
    plt.xlabel("Baseline Difficulty (Low Context)")
    plt.ylabel("Reduction in Uncertainty (Info Gain)")
    plt.axhline(0, color='grey', linestyle='--')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_df = run_experiment()
    # Remove NaN values (caused by empty bins in empirical calc)
    results_df = results_df.dropna()
    visualize_results(results_df)


# if __name__ == "__main__":
#     final_score, df = mock_data(N_SAMPLES)
#     print(df.head(30))
