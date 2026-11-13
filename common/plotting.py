import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator



def set_neurips_style():
    sns.set_theme(style="white", context="paper", font="DejaVu Sans")

    mpl.rcParams.update({
        # DPI Settings
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,

        # Typography (NeurIPS text width: 5.5 inches)
        "font.size": 8+2,
        "axes.titlesize": 8.5+2,
        "axes.labelsize": 7.5+2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,

        # Axes & Lines
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,

        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,

        # Vector / PDF font embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    })


def plot_line(df, configs):
    epochs = range(len(df[df.columns[0]]))
    
    for config in configs:
        sns.lineplot(df, x=epochs, y=config["column"], marker="o", **config["lineplot_kwargs"])
        config["lineplot_kwargs"]["ax"].set_title(config["title"])
        config["lineplot_kwargs"]["ax"].set_xlabel(config["xlabel"])
        config["lineplot_kwargs"]["ax"].set_ylabel(config["ylabel"], **config["ylabel_kwargs"])
        
        if "ylim" in config:
            config["lineplot_kwargs"]["ax"].set_ylim((config["ylim"]))

        config["lineplot_kwargs"]["ax"].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        sns.despine(ax=config["lineplot_kwargs"]["ax"], left=False, bottom=False)

def plot_distrib(df, configs):
    for config in configs:
        sns.histplot(df, x=config["column"], **config["dist_kwargs"])
        config["dist_kwargs"]["ax"].set_title(config["title"])
        config["dist_kwargs"]["ax"].set_xlabel(config["xlabel"])
        config["dist_kwargs"]["ax"].set_ylabel(config["ylabel"])
        
        if "ylim" in config:
            config["dist_kwargs"]["ax"].set_ylim((config["ylim"]))

        config["dist_kwargs"]["ax"].xaxis.set_major_locator(MaxNLocator(nbins=config["nbins"] if "nbins" in config else 10, integer=True))
        
        sns.despine(ax=config["dist_kwargs"]["ax"], left=False, bottom=False)


def plot_bar(values, categories, hue=None, title=None, xlabel="", ylabel="", fmt="%.0f%%", figsize=(3.25, 2.4), color=None, fig=None, ax=None):
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize)

    if hue is None:
        hue = categories

    if color is None:
        sns.barplot(x=categories, y=values, palette="colorblind", hue=hue, legend=False, ax=ax)
    else:
        if len(color) > 1:
            sns.barplot(x=categories, y=values, palette=color, hue=categories, legend=False, ax=ax)
        else:
            sns.barplot(x=categories, y=values, color=color, legend=False, ax=ax)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=fmt,
            padding=2,
            fontsize=7,
            color="#333333",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.grid(axis="x", visible=False)

    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()

    return fig, ax
