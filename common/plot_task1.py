import json
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from common.plotting import set_neurips_style, plot_line, plot_bar


base = f"checkpoints/task1"
results_base = Path("results/task1/")

dfs = {}
summaries = {}
for model in os.listdir(base):
    path = f"{base}/{model}/training_summary.json"
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data["history"])

    summaries[model] = data["summary"]
    dfs[model] = df


print("Metrics: ", list(dfs["l2_sp"].columns))
metrics = ["sft_loss"]

set_neurips_style()
f, ax = plt.subplots(1, 2, figsize=(2.0*3.25, 1.5*2.7), layout="constrained")
ax = np.atleast_1d(ax).ravel()
window_size = 30

for m, a in zip(metrics, ax):
    print("Plotting: ", m)
    for model in dfs:
        df = dfs[model]

        smoothed_col = f"{m}_smoothed"
        df[smoothed_col] = df[m].rolling(window=window_size, min_periods=1).mean()
        # df.plot(x="step", y=smoothed_col, ax=a, linewidth=2, legend=True, label=model.upper().replace("_", " "))
        
        # a.set_title(m)
    
        cfg = [
            {
                "lineplot_kwargs": {
                    "ax": a,
                    "markersize": 0,
                    "label": model.upper().replace("_", " ").replace("LORA", "LoRA"),
                },
                "title": "",
                "xlabel": "Step",
                "ylabel": "SFT Loss",
                "ylabel_kwargs": {},
                "column": smoothed_col,
            }
        ]
        plot_line(df, cfg)

# handles, labels = ax[0].get_legend_handles_labels()
# f.legend(
#     handles, labels, 
#     loc="upper center", 
#     bbox_to_anchor=(0.5, 1.04), 
#     ncols=3, 
#     frameon=True
# )

values = [summaries["full_ft"]["model_displacement_l2"]+1, summaries["l2_sp"]["model_displacement_l2"]]
categories = ["Full FT", "Full FT + L2-SP"]
plot_bar(values, categories, xlabel="", ylabel="Displacement", fmt="%.4f", figsize=(3.25, 2.4), color=None, fig=f, ax=ax[1])

plt.savefig(results_base / "loss.pdf")
