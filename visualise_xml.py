#!/usr/bin/env python3
import argparse
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# XML → pandas helpers
# -----------------------------

def parse_xml_to_df(file_path: str) -> pd.DataFrame:
    """Parse a PRESTO-style XML file of candidates into a DataFrame.

    Expected fields per <candidate>:
      period (float), dm (float), snr (float), nassoc (int), acc (float), nh (int)
    """
    doc = ET.parse(file_path)
    root = doc.getroot()

    periods = [float(x.text) for x in root.findall("candidates/candidate/period")]
    dms = [float(x.text) for x in root.findall("candidates/candidate/dm")]
    snr = [float(x.text) for x in root.findall("candidates/candidate/snr")]
    nassoc = [int(x.text) for x in root.findall("candidates/candidate/nassoc")]
    acc = [float(x.text) for x in root.findall("candidates/candidate/acc")]
    nh = [int(x.text) for x in root.findall("candidates/candidate/nh")]

    df = pd.DataFrame({
        "period": periods,
        "dm": dms,
        "snr": snr,
        "nassoc": nassoc,
        "acc": acc,
        "nh": nh,
        "source_file": Path(file_path).name,
    })
    return df


def load_many_xml(paths_or_globs: list[str]) -> pd.DataFrame:
    """Load many XML files (globs allowed) to one DataFrame."""
    all_paths = []
    for item in paths_or_globs:
        expanded = glob.glob(item)
        if not expanded:
            # Treat as literal path if glob doesn't match (so argparse completion works)
            expanded = [item]
        all_paths.extend(expanded)

    if not all_paths:
        raise SystemExit("No XML files found. Check your paths or globs.")

    frames = []
    for p in sorted(set(all_paths)):
        try:
            frames.append(parse_xml_to_df(p))
        except Exception as e:
            print(f"[warn] Skipping {p}: {e}")
    if not frames:
        raise SystemExit("No valid XML files parsed.")
    return pd.concat(frames, ignore_index=True)


# -----------------------------
# Plotters (each returns an Axes)
# -----------------------------

def plot_period_vs_snr(ax, df, *, color, alpha, period_vline=None):
    ax.scatter(df["period"], df["snr"], c=color, alpha=alpha)
    ax.set_xlabel("period")
    ax.set_ylabel("snr")
    if period_vline is not None:
        ax.axvline(x=period_vline, color="r", linestyle="--")
    return ax


def plot_dm_vs_snr(ax, df, *, color, alpha):
    ax.scatter(df["dm"], df["snr"], c=color, alpha=alpha)
    ax.set_xlabel("dm")
    ax.set_ylabel("snr")
    return ax


def plot_acc_vs_snr(ax, df, *, color, alpha):
    ax.scatter(df["acc"], df["snr"], c=color, alpha=alpha)
    ax.set_xlabel("acc")
    ax.set_ylabel("snr")
    return ax


def plot_nassoc_vs_snr(ax, df, *, color, alpha):
    ax.scatter(df["nassoc"], df["snr"], c=color, alpha=alpha)
    ax.set_xlabel("nassoc")
    ax.set_ylabel("snr")
    return ax


def plot_nh_vs_snr(ax, df, *, color, alpha):
    ax.scatter(df["nh"], df["snr"], c=color, alpha=alpha)
    ax.set_xlabel("nh")
    ax.set_ylabel("snr")
    return ax


def plot_dm_vs_period_bubble(ax, df, *, color, alpha, snr_scale):
    scatter = ax.scatter(df["dm"], df["period"], c=color, alpha=alpha, s=df["snr"] * snr_scale)
    ax.set_xlabel("dm")
    ax.set_ylabel("period")
    # Legend for bubble sizes
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
    legend = ax.legend(handles, labels, loc="upper right", title="SNR")
    for txt in legend.get_texts():
        txt.set_color(color)
    if legend.get_title():
        legend.get_title().set_color(color)
    return ax


def plot_hist_period(ax, df, *, color, alpha, bins, period_xlim=None):
    ax.hist(df["period"], bins=bins, color=color, alpha=alpha)
    if period_xlim is not None:
        ax.set_xlim(*period_xlim)
    ax.set_xlabel("period")
    ax.set_ylabel("count")
    return ax


def plot_hist_dm(ax, df, *, color, alpha, bins):
    ax.hist(df["dm"], bins=bins, color=color, alpha=alpha)
    ax.set_xlabel("DM (pc/cm³)")
    ax.set_ylabel("count")
    return ax

def plot_hist_snr(ax, df, *, color, alpha, bins, snr_xlim=None):
    ax.hist(df["snr"], bins=bins, color=color, alpha=alpha)
    if snr_xlim is not None:
        ax.set_xlim(*snr_xlim)
    ax.set_xlabel("SNR")
    ax.set_ylabel("count")
    return ax


PLOTTERS = {
    "period_vs_snr": plot_period_vs_snr,
    "dm_vs_snr": plot_dm_vs_snr,
    "acc_vs_snr": plot_acc_vs_snr,
    "nassoc_vs_snr": plot_nassoc_vs_snr,
    "nh_vs_snr": plot_nh_vs_snr,
    "dm_vs_period_bubble": plot_dm_vs_period_bubble,
    "hist_period": plot_hist_period,
    "hist_dm": plot_hist_dm,
    "hist_snr": plot_hist_snr,
}

DEFAULT_PLOTS = [
    "period_vs_snr",
    "dm_vs_snr",
    "acc_vs_snr",
    "nassoc_vs_snr",
    "nh_vs_snr",
    "dm_vs_period_bubble",
    "hist_period",
    "hist_dm",
    "hist_snr",
]


# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Visualize candidate data from one or more XML files (pandas version). Author: Fazal. Version 12Aug2025")
    p.add_argument("xml_file", nargs="+", type=str, help="Path(s) or glob(s) to XML file(s)")

    # Filtering
    p.add_argument("--period-max", type=float, default=0.01, help="Keep rows with period < PERIOD_MAX (default: 0.01)")
    p.add_argument("--min-snr", type=float, default=None, help="Optional minimum SNR filter")

    # Plot selection
    p.add_argument(
        "--plots",
        type=str,
        nargs="+",
        choices=sorted(PLOTTERS.keys()),
        default=DEFAULT_PLOTS,
        help="Which plots to include (default: all). Order is respected.",
    )
    p.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        choices=sorted(PLOTTERS.keys()),
        default=[],
        help="Plots to exclude (overrides --plots)",
    )

    # Figure + styling
    p.add_argument("--figwidth", type=float, default=20.0, help="Figure width in inches (default: 20)")
    p.add_argument("--per_plot_height", type=float, default=3.0, help="Height per subplot in inches (default: 3)")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    p.add_argument("--alpha", type=float, default=0.5, help="Point/bar alpha (default: 0.5)")
    p.add_argument("--color", type=str, default="purple", help="Matplotlib color for points/bars (default: purple)")
    p.add_argument("--title", type=str, default="Candidates Overview", help="Figure title")

    # Plot-specific tuning
    p.add_argument("--snr_scale", type=float, default=1.0, help="Bubble size scale for dm_vs_period_bubble (default: 1.0)")
    p.add_argument("--period_hist_bins", type=int, default=2000, help="Bins for period histogram (default: 2000)")
    p.add_argument("--dm_hist_bins", type=int, default=20, help="Bins for DM histogram (default: 20)")
    p.add_argument("--snr_hist_bins", type=int, default=1000, help="Bins for SNR histogram (default:1000)")
    p.add_argument("--snr_hist_xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"), help="Optional xlim for snr histogram and/or snr axis")
    p.add_argument("--period_xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"), help="Optional xlim for period histogram and/or period axis")
    p.add_argument("--period_vline", type=float, default=None, help="Optional vertical line in period vs snr plot")

    # Output
    p.add_argument("--outfile", type=str, default="xml_statistics.png", help="Output image filename (default: xml_statistics.png)")

    return p


# -----------------------------
# Main
# -----------------------------

def main():
    args = build_argparser().parse_args()

    df = load_many_xml(args.xml_file)

    # Filters
    if args.period_max is not None:
        df = df[df["period"] < args.period_max]
    if args.min_snr is not None:
        df = df[df["snr"] >= args.min_snr]

    # Determine plot list
    plots = [p for p in args.plots if p in PLOTTERS]
    if args.exclude:
        plots = [p for p in plots if p not in set(args.exclude)]
    if not plots:
        raise SystemExit("No plots selected after applying --exclude.")

    # Figure size adapts to number of plots
    n = len(plots)
    fig_height = max(args.per_plot_height * n, 3)
    fig, axes = plt.subplots(n, 1, figsize=(args.figwidth, fig_height), dpi=args.dpi)
    if n == 1:
        axes = [axes]

    # Draw plots in requested order
    for ax, key in zip(axes, plots):
        fn = PLOTTERS[key]
        if key == "period_vs_snr":
            fn(ax, df, color=args.color, alpha=args.alpha, period_vline=args.period_vline)
            if args.period_xlim is not None:
                ax.set_xlim(*args.period_xlim)
        elif key == "dm_vs_period_bubble":
            fn(ax, df, color=args.color, alpha=args.alpha, snr_scale=args.snr_scale)
        elif key == "hist_period":
            fn(ax, df, color=args.color, alpha=args.alpha, bins=args.period_hist_bins, period_xlim=args.period_xlim)
        elif key == "hist_dm":
            fn(ax, df, color=args.color, alpha=args.alpha, bins=args.dm_hist_bins)
        elif key == "hist_snr":
            fn(ax, df, color=args.color, alpha=args.alpha, bins=args.snr_hist_bins, snr_xlim=args.snr_hist_xlim)
        else:
            # Generic signatures
            fn(ax, df, color=args.color, alpha=args.alpha)

    plt.tight_layout()
    fig.suptitle(args.title)
    # suptitle can overlap; adjust a bit
    fig.subplots_adjust(top=0.95)

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"Saved figure to: {out.resolve()}")


if __name__ == "__main__":
    main()

