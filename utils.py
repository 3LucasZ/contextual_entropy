import matplotlib.pyplot as plt
import pandas as pd
import re


def setupPlt():
    import scienceplots
    plt.style.use(["science"])
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['figure.titlesize'] = 15


color_map = {
    "[]": "lightgray",  # Baseline/No Context

    "['continent']": "red",
    "['market_cap']": "blue",
    "['sector']": "gold",

    "['continent', 'market_cap']": "purple",  # Red + Blue
    "['continent', 'sector']": "orange",      # Red + Yellow
    "['market_cap', 'sector']": "green",      # Blue + Yellow

    "['continent', 'market_cap', 'sector']": "sienna"  # Mix of all three
}

# color_map = {
#     "[]": "#e0e0e0",  # Light Grey

#     # Primary Sets
#     "['continent']": "#00FFFF",  # Cyan
#     "['market_cap']": "#FF00FF",  # Magenta
#     "['sector']": "#FFD700",     # Yellow (Gold)

#     # Intersections (Resulting primaries)
#     "['continent', 'market_cap']": "#0000FF",  # Blue
#     "['continent', 'sector']": "#00FF00",     # Green (Lime)
#     "['market_cap', 'sector']": "#FF0000",    # Red

#     # All Three
#     "['continent', 'market_cap', 'sector']": "#333333"  # Dark Charcoal
# }

# color_map = {
#     "[]": "#d3d3d3",                          # Gray (0 items)

#     "['continent']": "#a6cee3",               # Light Blue
#     "['market_cap']": "#1f78b4",              # Medium Blue
#     "['sector']": "#17becf",                  # Cyan

#     "['continent', 'market_cap']": "#b2df8a",  # Light Green
#     "['continent', 'sector']": "#33a02c",     # Medium Green
#     "['market_cap', 'sector']": "#8dd3c7",    # Teal

#     # Distinct Purple (3 items)
#     "['continent', 'market_cap', 'sector']": "#7570b3"
# }


def extract_info(row):
    trial = row['Trial'][2:-2]
    prompt = row['Prompt']

    # Identify Group & Strata via Regex
    if trial == "continent":
        group = "Continent"
        match = re.search(r"The stock is in (.*?)\. Will", prompt)
        strata = match.group(1) if match else "Unknown"

    elif trial == "market_cap":
        group = "Market Cap"
        match = re.search(r"The stock is a (.*?) company\. Will", prompt)
        strata = match.group(1) if match else "Unknown"

    elif trial == "sector":
        group = "Sector"
        match = re.search(r"The stock is in the (.*?) sector\. Will", prompt)
        strata = match.group(1) if match else "Unknown"
        strata = strata.replace(" ", "\n")
        if "Info" in strata:
            strata = "Info Tech"
        if "Discretionary" in strata:
            strata = "Consumer Disc"
        if "Communication" in strata:
            strata = "Comms Services"
    else:
        group = "ERROR"
        strata = "ERROR"
    return pd.Series([group, strata])
