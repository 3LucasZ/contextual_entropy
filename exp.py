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

# -- Regular --
engine = Engine(config)
engine.run_combinations()

# -- Qwen --
config["model"] = "qwen"
config["file"] = "yes_no_qwen"
engine = Engine(config)
engine.run_combinations()

# -- Qwen Neither --
# config["model"] = "qwen"
# config["neither"] = True
# config["file"] = "yes_no_or_qwen"
# engine = Engine(config)
# engine.run_combinations()

plotter = Plotter()
# plotter.plot("", "yes_no", "yes_no", False)
# plotter.plot_cat("", "yes_no", "yes_no_cat", False)
# plotter.plot("", "yes_no_or", "yes_no_or", False)
# plotter.plot_cat("", "yes_no_or", "yes_no_or_cat", False)
# plotter.plot("", "yes_no_qwen", "yes_no_qwen", False)
plotter.plot_cat("", "yes_no_qwen", "yes_no_qwen_cat", False)
# plotter.plot("", "yes_no_or_qwen", "yes_no_or_qwen", False)
plotter.plot_cat("", "yes_no_or_qwen", "yes_no_or_qwen_cat", False)
