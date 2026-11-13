import json
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from common.plotting import set_neurips_style, plot_line, plot_bar


base = Path("checkpoints/task2/")
results_base = Path("results/task2/")

with open(results_base / "dev" / "dev.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

cap_loss = []
cond_losses = {}
updates = []
for line in data:
    for condition in line:
        if condition == "caption":
            cap_loss.append(line[condition])
        if condition == "main_run_updates":
            updates.append(line[condition])
        if condition in ["main_run_updates", "caption"]: continue

        if condition not in cond_losses:
            cond_losses[condition] = {}
        ds_metrics = line[condition]
        for ds in ds_metrics:
            if ds not in cond_losses[condition]:
                cond_losses[condition][ds] = []
            cond_losses[condition][ds].append(ds_metrics[ds])

# for cond in cond_losses:
#     cond_losses[cond] = pd.DataFrame(cond_losses[cond])

print(list(cond_losses.keys()))

set_neurips_style()
f, ax = plt.subplots(2, 3, figsize=(2.5*3.25, 1.0*2.7), layout="constrained")
ax = np.atleast_1d(ax).ravel()

ax[0].plot(cap_loss)
ax[0].set_ylabel("Caption Loss")

for cond, a in zip(cond_losses, ax[1:]):
    for ds in cond_losses[cond]:
        a.plot(cond_losses[cond][ds], label=ds)
    a.set_ylabel(cond.title())
    a.legend()

for a in ax:
    for (i, b), c in zip(enumerate([2250, 2250+1500, 3750+750]), ["red", "green", "blue"]):
        a.axvline(x=b//500, color=c, linestyle='--', linewidth=1.5, label=f"Stage-{i+1}")

ax[-1].remove()

plt.savefig(results_base / "dev-loss.pdf")
