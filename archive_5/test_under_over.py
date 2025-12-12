from engine import Engine
from plotter import Plotter


config = {
    "model": "llama",
    "system": "You are a professional stock analyst and trader. You must answer with only Underperform or Overperform.",
    "question": "Will this stock underperform or overperform the median over the next 12 months?",
    "token1": "Under",
    "token2": "Over",
    "neither": False,
    "file": "under_over",
    "verbose": True,
    "quick": True,
}
plotter = Plotter()

# -- Regular --
engine = Engine(config)
engine.run_combinations()
plotter.plot("", "under_over", "under_over", False)
plotter.plot_cat("", "under_over", "under_over_cat", False)


# -- Neither --
config["neither"] = True
config["file"] = "under_over_or"
engine = Engine(config)
engine.run_combinations()
plotter.plot("", "under_over_or", "under_over_or", False)
plotter.plot_cat("", "under_over_or", "under_over_or_cat", False)

# -- Qwen --
config["model"] = "qwen"
config["neither"] = False
config["file"] = "under_over_qwen"
engine = Engine(config)
engine.run_combinations()
plotter.plot("", "under_over_qwen", "under_over_qwen", False)
plotter.plot_cat("", "under_over_qwen", "under_over_qwen_cat", False)

# -- Qwen Neither --
config["model"] = "qwen"
config["neither"] = True
config["file"] = "under_over_or_qwen"
engine = Engine(config)
engine.run_combinations()
plotter.plot("", "under_over_or_qwen", "under_over_or_qwen", False)
plotter.plot_cat("", "under_over_or_qwen", "under_over_or_qwen_cat", False)
