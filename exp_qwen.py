from engine import Engine
from plotter import Plotter


config = {
    "model": "qwen",
    "system": "You are a professional stock analyst and trader. Answer with only Underperform or Overperform.",
    "question": "Will this stock underperform or overperform the S&P 500 over the next 12 months?",
    "token1": "Under",
    "token2": "Over",
    "neither": False,
    "file": "under_over_qwen",
    "verbose": True,
    "quick": True,
}
plotter = Plotter()

# -- Qwen --
engine = Engine(config)
engine.run_combinations()

plotter.plot("", "under_over_qwen", "under_over_qwen", False)
plotter.plot_cat("", "under_over_qwen", "under_over_qwen_cat", False)
