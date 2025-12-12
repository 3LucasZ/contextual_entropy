from engine import Engine
from plotter import Plotter


config = {
    "model": "llama",
    "system": "You are a professional stock analyst and trader. Answer Yes or No.",
    "question": "Will this stock underperform the median stock over the next 12 months?",
    "token1": "Yes",
    "token2": "No",
    "neither": False,
    "file": "yes_no",
    "verbose": True,
    "quick": True,
}
plotter = Plotter()

# -- Regular --
engine = Engine(config)
engine.run_combinations()

plotter.plot("", "yes_no", "yes_no", False)
plotter.plot_cat("", "yes_no", "yes_no_cat", False)

# -- Qwen --
config["model"] = "qwen"
config["file"] = "yes_no_qwen"
engine = Engine(config)
engine.run_combinations()

plotter.plot("", "yes_no_qwen", "yes_no_qwen", False)
plotter.plot_cat("", "yes_no_qwen", "yes_no_qwen_cat", False)
