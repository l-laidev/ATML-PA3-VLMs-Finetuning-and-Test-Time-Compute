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

steps = []
for stage in range(1, 3+1):
    path = base / f"stage-{stage}" / f"stage-{stage}"
    files = sorted(filter(lambda x: x.endswith(".json"), os.listdir(path)), key=lambda x: int(x.split(".")[0].split("_")[-1]))
    for fp in files:
        with open(path/fp, "r") as f:
            for line in json.load(f):
                if line["main_run_updates"] >= len(steps):
                    steps.append(line)
                else:
                    steps = steps[:line["main_run_updates"]]

df = {"loss": [], "stage": [], "update": []}
peak_vram = [0, 0, 0]
total_time = [0, 0, 0]
trainable = [0, 0, 0]
boundaries = [0, 0, 0]
for line in sorted(steps, key=lambda x: x["main_run_updates"]):
    df["loss"].append(line["loss"])
    df["stage"].append(line["stage"])
    df["update"].append(line["main_run_updates"])

    peak_vram[line["stage"]-1] = max(peak_vram[line["stage"]-1], line["peak_vram_mb"])
    total_time[line["stage"]-1] += line["duration"]
    trainable[line["stage"]-1] = line["parameters"]["connector"]["trainable"] + line["parameters"]["llm"]["trainable"]
    boundaries[line["stage"]-1] = max(boundaries[line["stage"]-1], line["main_run_updates"])

print(peak_vram, total_time, trainable, boundaries)

set_neurips_style()
f, ax = plt.subplots(1, 1, figsize=(2.0*3.25, 1.5*2.7), layout="constrained")
ax = np.atleast_1d(ax).ravel()

df = pd.DataFrame(df)
for m, a in zip(["loss"], ax):
    window_size = 30
    smoothed_col = f"{m}_smoothed"
    df[smoothed_col] = df[m].rolling(window=window_size, min_periods=1).mean()
    # df.plot(x="step", y=smoothed_col, ax=a, linewidth=2, legend=True, label=model.upper().replace("_", " "))

    # a.set_title(m)

    cfg = [
        {
            "lineplot_kwargs": {
                "ax": a,
                "markersize": 0,
            },
            "title": "",
            "xlabel": "Step",
            "ylabel": "SFT Loss",
            "ylabel_kwargs": {},
            "column": smoothed_col,
        }
    ]
    plot_line(df, cfg)
    for (i, b), c in zip(enumerate(boundaries), ["red", "green", "blue"]):
        a.axvline(x=b, color=c, linestyle='--', linewidth=1.5, label=f"Stage-{i+1}")
    a.legend()

plt.savefig(results_base / "loss.pdf")
