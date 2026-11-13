import json
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from common.plotting import set_neurips_style, plot_line, plot_bar


ds = "mathvista"
split = "dev"
results_base = Path("results/task3/") / ds
stored = results_base / "main" / split

lines = []
for res in os.listdir(stored):
    path = stored / res
    with open(path, "r") as f:
        data = json.load(f)
    lines.append(data)

# {"Budget": 1, "Avg-FirstAns": 0.7334999999999999, "SelectionFailure-FirstAns": 0.0, "SamplingFailure-FirstAns": 0.2665000000000001,
# "Avg-SelfConsist": 0.7334999999999999, "SelectionFailure-SelfConsist": 0.0, "SamplingFailure-SelfConsist": 0.2665000000000001,
# "Avg-Oracle": 0.7334999999999999, "Avg-Tokens": 577.7349999999999,
# "Avg-Highest-Logp": 0.7334999999999999, "SelectionFailure-Highest-Logp": 0.0, "SamplingFailure-Highest-Logp": 0.2665000000000001,
# "Avg-Highest-SelfCert": 0.7334999999999999, "SelectionFailure-Highest-SelfCert": 0.0, "SamplingFailure-Highest-SelfCert": 0.2665000000000001,
# "Avg-Highest-SelfCert-RWC": 0.7334999999999999, "SelectionFailure-Highest-SelfCert-RWC": 0.0, "SamplingFailure-Highest-SelfCert-RWC": 0.2665000000000001,
# "Avg-DistinctGroups": 1.0}

accs = {}
sel_fails = {}
sam_fails = {}
oracle = {}
assocs = {}
for budget in lines:
    b = budget["Budget"]
    for k in budget.keys():
        if any([kw in k for kw in ["Tokens", "DistinctGroup"]]): continue

        if "Association" in k:
            k2 = k.replace("Avg-", "")
            if k2 not in assocs: assocs[k2] = []
            
            assocs[k2].append(budget[k])
        elif k.startswith("Avg-Oracle"):
            k2 = k.replace("Avg-", "")
            if k2 not in oracle: oracle[k2] = []
            
            oracle[k2].append(budget[k])
        elif k.startswith("Avg-"):
            k2 = k.replace("Avg-", "")
            if k2 not in accs: accs[k2] = []
            
            accs[k2].append(budget[k])
        elif k.startswith("SelectionFailure"):
            k2 = k.replace("SelectionFailure-", "")
            if k2 not in sel_fails: sel_fails[k2] = []
            
            sel_fails[k2].append(budget[k])
        elif k.startswith("SamplingFailure"):
            k2 = k.replace("SamplingFailure-", "")
            if k2 not in sam_fails: sam_fails[k2] = []
            
            sam_fails[k2].append(budget[k])

set_neurips_style()
f, ax = plt.subplots(2, 2, figsize=(2.25*3.25, 1.5*2.7), layout="constrained")
ax = np.atleast_1d(ax).ravel()

accs = pd.DataFrame(accs)
sel_fails = pd.DataFrame(sel_fails)
sam_fails = pd.DataFrame(sam_fails)
oracle = pd.DataFrame(oracle)

for method in accs.columns:
    for a, ylabel, data in zip(
        ax,
        ["Accuracy", "Selection Failure", "Sampling Failure"],
        [accs, sel_fails, sam_fails],
    ):
        if method not in data.columns: continue
        
        cfg = [
            {
                "lineplot_kwargs": {
                    "ax": a,
                    "markersize": 0,
                    "label": method,
                },
                "title": "",
                "xlabel": "Budget",
                "ylabel": ylabel,
                "ylabel_kwargs": {},
                "column": method,
                "ylim": (0, 1),
            }
        ]
        plot_line(data, cfg)

cfg = [
    {
        "lineplot_kwargs": {
            "ax": ax[3],
            "markersize": 0,
            # "label": "",
        },
        "title": "",
        "xlabel": "Budget",
        "ylabel": "Oracle Coverage",
        "ylabel_kwargs": {},
        "column": oracle.columns[0],
        "ylim": (0, 1),
    }
]
plot_line(oracle, cfg)

handles, labels = ax[0].get_legend_handles_labels()
for a in ax:
    legend = a.get_legend()
    if legend is not None:
        legend.remove()
f.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

plt.savefig(results_base / f"{ds}-curves.pdf")


print(assocs)

# for m, a in zip(metrics, ax):
#     print("Plotting: ", m)
#     for model in dfs:
#         df = dfs[model]

#         smoothed_col = f"{m}_smoothed"
#         df[smoothed_col] = df[m].rolling(window=window_size, min_periods=1).mean()
#         # df.plot(x="step", y=smoothed_col, ax=a, linewidth=2, legend=True, label=model.upper().replace("_", " "))
        
#         # a.set_title(m)
    
#         cfg = [
#             {
#                 "lineplot_kwargs": {
#                     "ax": a,
#                     "markersize": 0,
#                     "label": model.upper().replace("_", " ").replace("LORA", "LoRA"),
#                 },
#                 "title": "",
#                 "xlabel": "Step",
#                 "ylabel": "SFT Loss",
#                 "ylabel_kwargs": {},
#                 "column": smoothed_col,
#             }
#         ]
#         plot_line(df, cfg)

# # handles, labels = ax[0].get_legend_handles_labels()
# # f.legend(
# #     handles, labels, 
# #     loc="upper center", 
# #     bbox_to_anchor=(0.5, 1.04), 
# #     ncols=3, 
# #     frameon=True
# # )

# values = [summaries["full_ft"]["model_displacement_l2"]+1, summaries["l2_sp"]["model_displacement_l2"]]
# categories = ["Full FT", "Full FT + L2-SP"]
# plot_bar(values, categories, xlabel="", ylabel="Displacement", fmt="%.4f", figsize=(3.25, 2.4), color=None, fig=f, ax=ax[1])

# plt.savefig(results_base / "loss.pdf")
