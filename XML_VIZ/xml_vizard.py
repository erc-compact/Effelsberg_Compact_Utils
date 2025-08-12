#!/usr/bin/env python3
"""
XML Candidate Visualizer (pandas)

Discovery-focused plots:
- pdm_hex: Period–DM hexbin (SNR-weighted, optional log-period)
- pacc_scatter: Period–Acceleration scatter (size/color by SNR)
- dmacc_hex: DM–Acceleration hexbin (SNR-weighted)
- harmonic_overlay: Overlay fundamental + harmonics from top-K candidates
- fundamental_hist: Histogram of “fundamentalized” periods (collapses harmonics)
- nassoc_snr: nassoc vs SNR (colored by DM) with multi-beam guideline
- snr_ecdf: Empirical CDF of SNR with threshold marker
- facet_p_snr_by_dm: Small multiples of Period–SNR by DM bands
- snr_by_dm_ridgeline: SNR distributions by DM bins (ridgeline/joy)
- kde_pdm: KDE contours on Period–DM (fallback to smoothed 2D hist if SciPy missing)
- hist_snr: SNR histogram (bin width & x-lims configurable)
- rfi_landmarks: Optional RFI landmarks on period/DM axes (mains/clock & DM≈0 band)

Tip: run with --list-plots to see all plot keys.
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# XML → pandas
# -----------------------------

def parse_xml_to_df(file_path: str) -> pd.DataFrame:
    import xml.etree.ElementTree as ET
    doc = ET.parse(file_path)
    root = doc.getroot()
    periods = [float(x.text) for x in root.findall("candidates/candidate/period")]
    dms = [float(x.text) for x in root.findall("candidates/candidate/dm")]
    snr = [float(x.text) for x in root.findall("candidates/candidate/snr")]
    nassoc = [int(x.text) for x in root.findall("candidates/candidate/nassoc")]
    acc = [float(x.text) for x in root.findall("candidates/candidate/acc")]
    nh = [int(x.text) for x in root.findall("candidates/candidate/nh")]
    return pd.DataFrame({
        "period": periods,
        "dm": dms,
        "snr": snr,
        "nassoc": nassoc,
        "acc": acc,
        "nh": nh,
        "source_file": Path(file_path).name,
    })


def load_many_xml(paths_or_globs: list[str]) -> pd.DataFrame:
    all_paths = []
    for pat in paths_or_globs:
        matches = glob.glob(pat)
        if matches:
            all_paths.extend(matches)
        else:
            all_paths.append(pat)
    if not all_paths:
        raise SystemExit("No XML files found.")
    frames = []
    for p in sorted(set(all_paths)):
        try:
            frames.append(parse_xml_to_df(p))
        except Exception as e:
            print(f"[warn] Skipping {p}: {e}")
    if not frames:
        raise SystemExit("No valid XML parsed.")
    return pd.concat(frames, ignore_index=True)

# -----------------------------
# Plot utilities
# -----------------------------

def add_rfi_landmarks(ax, periods=None, dm_bands=None):
    if periods:
        for p in periods:
            ax.axvline(p, ls='--', lw=1, alpha=0.6)
    if dm_bands and len(dm_bands) == 2:
        lo, hi = min(dm_bands), max(dm_bands)
        ax.axvspan(lo, hi, color='gray', alpha=0.1)

def hexbin(ax, x, y, C=None, gridsize=60, xlabel='', ylabel='', title=''):
    if C is not None:
        hb = ax.hexbin(x, y, C=C, gridsize=gridsize, reduce_C_function=np.sum)
        cbar_label = 'SNR sum'
    else:
        hb = ax.hexbin(x, y, gridsize=gridsize)
        cbar_label = 'count'
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    cbar = plt.colorbar(hb, ax=ax); cbar.set_label(cbar_label)

def kde2d(ax, x, y, levels=(0.5, 0.8, 0.95), gridsize=200, xlabel='', ylabel='', title=''):
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        # Fallback: smoothed 2D histogram contours
        H, xedges, yedges = np.histogram2d(x, y, bins=80)
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=1.2)
        except Exception:
            pass
        X = 0.5*(xedges[:-1]+xedges[1:]); Y = 0.5*(yedges[:-1]+yedges[1:])
        ax.contour(X, Y, H.T, levels=6)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title + " (hist fallback)")
        return
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    XX, YY = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    cs = ax.contour(XX, YY, ZZ, levels=np.quantile(ZZ, list(levels)))
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)

def ecdf(ax, values, xlabel='SNR', title='SNR ECDF'):
    v = np.sort(values)
    y = np.arange(1, len(v)+1)/len(v)
    ax.plot(v, y)
    ax.set_xlabel(xlabel); ax.set_ylabel('F(x)'); ax.set_title(title); ax.grid(True)

def fundamentalize_periods(df, max_harm=8, rel_tol=1e-3):
    df_sorted = df.sort_values('snr', ascending=False).reset_index(drop=True)
    periods = df_sorted['period'].values.copy()
    snr = df_sorted['snr'].values
    used = periods.copy()
    for i in range(len(periods)):
        p = periods[i]
        for n in range(2, max_harm+1):
            cand = p * n
            j = np.where(np.isclose(used, cand, rtol=rel_tol))[0]
            if j.size > 0 and snr[j[0]] >= snr[i]:
                periods[i] = cand
                break
    return periods

# -----------------------------
# Plotters
# -----------------------------

def plot_pdm_hex(ax, df, *, log_period, gridsize, rfi_periods, rfi_dm_bands):
    x = np.log10(df['period']) if log_period else df['period']
    y = df['dm']
    hexbin(ax, x, y, C=df['snr'], gridsize=gridsize,
           xlabel=('log10(period)' if log_period else 'period'), ylabel='DM',
           title='P–DM hexbin (SNR-weighted)')
    add_rfi_landmarks(ax, periods=(np.log10(rfi_periods) if (rfi_periods and log_period) else rfi_periods),
                      dm_bands=rfi_dm_bands)

def plot_pacc_scatter(ax, df, *, size_scale, alpha):
    s = np.clip(df['snr']*size_scale, 5, None)
    sc = ax.scatter(df['period'], df['acc'], c=df['snr'], s=s, alpha=alpha)
    ax.set_xlabel('period'); ax.set_ylabel('acc'); ax.set_title('P–acc (color=SNR, size~SNR)')
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('SNR')

def plot_dmacc_hex(ax, df, *, gridsize):
    hexbin(ax, df['dm'], df['acc'], C=df['snr'], gridsize=gridsize,
           xlabel='DM', ylabel='acc', title='DM–acc hexbin (SNR-weighted)')

def plot_harmonic_overlay(ax, df, *, topk, max_harm, alpha):
    ax.scatter(df['period'], df['snr'], s=8, alpha=0.5)
    tops = df.nlargest(topk, 'snr')
    for _, row in tops.iterrows():
        p0 = row['period']
        ax.axvline(p0, color='tab:red', lw=1.2, alpha=alpha)
        for n in range(2, max_harm+1):
            ax.axvline(p0/n, color='tab:orange', lw=0.8, alpha=alpha*0.8)
    ax.set_xlabel('period'); ax.set_ylabel('SNR'); ax.set_title(f'Harmonic overlay (top {topk})')

def plot_fundamental_hist(ax, df, *, max_harm, bins):
    fund = fundamentalize_periods(df[['period','snr']].copy(), max_harm=max_harm)
    ax.hist(fund, bins=bins)
    ax.set_xlabel('fundamentalized period'); ax.set_ylabel('count'); ax.set_title('Fundamentalized period histogram')

def plot_nassoc_snr(ax, df, *, nbeams, snr_thresh):
    sc = ax.scatter(df['nassoc'], df['snr'], c=df['dm'], s=10, alpha=0.6)
    ax.set_xlabel('nassoc'); ax.set_ylabel('SNR'); ax.set_title('nassoc vs SNR (color=DM)')
    if nbeams:
        ax.axvline(max(1, nbeams//3), ls='--', alpha=0.7)
    if snr_thresh:
        ax.axhline(snr_thresh, ls='--', alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('DM')

def plot_snr_ecdf(ax, df, *, snr_thresh):
    ecdf(ax, df['snr'].values, xlabel='SNR', title='SNR ECDF')
    if snr_thresh:
        ax.axvline(snr_thresh, ls='--', alpha=0.7)

def plot_facet_p_snr_by_dm_into_slot(fig, slot, df, *, dm_edges):
    """Draw a facet grid *inside* a reserved subplot slot using SubGridSpec."""
    n = len(dm_edges) - 1
    rows = int(np.ceil(n / 2))
    subgs = slot.subgridspec(rows, 2)
    axes = []
    for i in range(n):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(subgs[r, c])
        lo, hi = dm_edges[i], dm_edges[i+1]
        sub = df[(df['dm'] >= lo) & (df['dm'] < hi)]
        ax.scatter(sub['period'], sub['snr'], s=8, alpha=0.6)
        ax.set_title(f'DM [{lo}, {hi})')
        ax.set_xlabel('period'); ax.set_ylabel('SNR')
        axes.append(ax)
    return rows, axes  # rows needed for height calculation

def plot_snr_by_dm_ridgeline(ax, df, *, dm_edges):
    offsets = np.arange(len(dm_edges)-1)
    for i, (lo, hi) in enumerate(zip(dm_edges[:-1], dm_edges[1:])):
        sub = df[(df['dm']>=lo) & (df['dm']<hi)]['snr'].values
        if sub.size == 0:
            continue
        hist, edges = np.histogram(sub, bins=min(50, max(10, sub.size//10)), density=True)
        centers = 0.5*(edges[:-1]+edges[1:])
        y = hist/np.max(hist) if np.max(hist)>0 else hist
        ax.plot(centers, y + i, lw=1)
        ax.fill_between(centers, i, y + i, alpha=0.3)
    ax.set_xlabel('SNR'); ax.set_yticks(offsets)
    ax.set_yticklabels([f"[{lo},{hi})" for lo,hi in zip(dm_edges[:-1], dm_edges[1:])])
    ax.set_title('SNR ridgeline by DM bin')

def plot_kde_pdm(ax, df, *, log_period):
    x = np.log10(df['period']) if log_period else df['period']
    y = df['dm']
    kde2d(ax, x, y, xlabel=('log10(period)' if log_period else 'period'),
          ylabel='DM', title='P–DM KDE contours')

# Legacy/utility
def plot_period_vs_snr(ax, df, *, color, alpha, period_vline=None):
    ax.scatter(df['period'], df['snr'], c=color, alpha=alpha)
    ax.set_xlabel('period'); ax.set_ylabel('snr')
    if period_vline is not None:
        ax.axvline(x=period_vline, color='r', linestyle='--')

def plot_dm_vs_snr(ax, df, *, color, alpha):
    ax.scatter(df['dm'], df['snr'], c=color, alpha=alpha)
    ax.set_xlabel('dm'); ax.set_ylabel('snr')

def plot_acc_vs_snr(ax, df, *, color, alpha):
    ax.scatter(df['acc'], df['snr'], c=color, alpha=alpha)
    ax.set_xlabel('acc'); ax.set_ylabel('snr')

def plot_nassoc_vs_snr(ax, df, *, color, alpha):
    ax.scatter(df['nassoc'], df['snr'], c=color, alpha=alpha)
    ax.set_xlabel('nassoc'); ax.set_ylabel('snr')

def plot_nh_vs_snr(ax, df, *, color, alpha):
    ax.scatter(df['nh'], df['snr'], c=color, alpha=alpha)
    ax.set_xlabel('nh'); ax.set_ylabel('snr')

def plot_dm_vs_period_bubble(ax, df, *, color, alpha, snr_scale):
    sc = ax.scatter(df['dm'], df['period'], c=df['snr'], alpha=alpha, s=df['snr']*snr_scale)
    ax.set_xlabel('dm'); ax.set_ylabel('period')
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label('SNR')

def plot_hist_period(ax, df, *, color, alpha, bins, period_xlim=None):
    ax.hist(df['period'], bins=bins, color=color, alpha=alpha)
    if period_xlim is not None:
        ax.set_xlim(*period_xlim)
    ax.set_xlabel('period'); ax.set_ylabel('count')

def plot_hist_dm(ax, df, *, color, alpha, bins):
    ax.hist(df['dm'], bins=bins, color=color, alpha=alpha)
    ax.set_xlabel('DM (pc/cm³)'); ax.set_ylabel('count')

def plot_snr_hist(ax, df, *, bin_size=2000, xlims=None):
    vals = df['snr'].values
    start = 0 if np.min(vals) >= 0 else np.floor(np.min(vals) / bin_size) * bin_size
    bins = np.arange(start, vals.max() + bin_size, bin_size)
    ax.hist(vals, bins=bins, alpha=0.7)
    ax.set_xlabel('SNR'); ax.set_ylabel('Count'); ax.set_title('SNR Histogram')
    if xlims:
        ax.set_xlim(xlims)

# Registry
PLOTTERS = {
    'period_vs_snr': plot_period_vs_snr,
    'dm_vs_snr': plot_dm_vs_snr,
    'acc_vs_snr': plot_acc_vs_snr,
    'nassoc_vs_snr': plot_nassoc_vs_snr,
    'nh_vs_snr': plot_nh_vs_snr,
    'dm_vs_period_bubble': plot_dm_vs_period_bubble,
    'hist_period': plot_hist_period,
    'hist_dm': plot_hist_dm,
    'hist_snr': plot_snr_hist,
    # new ones
    'pdm_hex': plot_pdm_hex,
    'pacc_scatter': plot_pacc_scatter,
    'dmacc_hex': plot_dmacc_hex,
    'harmonic_overlay': plot_harmonic_overlay,
    'fundamental_hist': plot_fundamental_hist,
    'nassoc_snr': plot_nassoc_snr,
    'snr_ecdf': plot_snr_ecdf,
    'facet_p_snr_by_dm': None,  # handled specially (subgrid)
    'snr_by_dm_ridgeline': plot_snr_by_dm_ridgeline,
    'kde_pdm': plot_kde_pdm,
}

DEFAULT_PLOTS = [
    'period_vs_snr', 'dm_vs_snr', 'acc_vs_snr', 'nassoc_vs_snr', 'nh_vs_snr', 'dm_vs_period_bubble', 'hist_period', 'hist_dm', 'hist_snr',
    'pdm_hex', 'kde_pdm', 'pacc_scatter', 'dmacc_hex', 'harmonic_overlay', 'fundamental_hist',
    'nassoc_snr', 'snr_ecdf', 'facet_p_snr_by_dm', 'snr_by_dm_ridgeline',
]

# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter,
                         argparse.RawTextHelpFormatter):
        pass

    epilog = (
        "Examples:\n"
        "  # Discovery bundle with KDE + RFI marks\n"
        "  viz_xml_candidates.py *.xml --plots pdm_hex kde_pdm harmonic_overlay fundamental_hist \\\n"
        "    --log-period --rfi-periods 1.0 0.02 0.0166667 --rfi-dm-bands 0 2\n\n"
        "  # Acceleration & RFI triage\n"
        "  viz_xml_candidates.py *.xml --plots pacc_scatter dmacc_hex nassoc_snr snr_ecdf \\\n"
        "    --nbeams 25 --snr-thresh 8\n\n"
        "  # Facets + ridgeline\n"
        "  viz_xml_candidates.py *.xml --plots facet_p_snr_by_dm snr_by_dm_ridgeline --dm-bins 0 20 100 300 1000\n\n"
        "  # Simple SNR histogram\n"
        "  viz_xml_candidates.py *.xml --plots hist_snr --snr-hist-bin 2000 --snr-hist-xlim 0 50000\n"
    )

    p = argparse.ArgumentParser(
        description='Visualize candidate data from XML (pandas), with pulsar-search focused plots.',
        formatter_class=SmartFormatter,
        epilog=epilog,
    )

    g_in = p.add_argument_group('Input')
    g_in.add_argument('xml_file', nargs='*', type=str, help='Path(s)/glob(s) to XML')

    g_filters = p.add_argument_group('Filters')
    g_filters.add_argument('--period-max', type=float, default=0.01, help='Keep rows with period < PERIOD_MAX')
    g_filters.add_argument('--min-snr', type=float, default=None, help='Optional minimum SNR filter')

    g_sel = p.add_argument_group('Plot selection')
    g_sel.add_argument('--plots', type=str, nargs='+', metavar='plot', default=DEFAULT_PLOTS,
                       help='Which plots to include (order respected). Use --list-plots to see all plot keys.')
    g_sel.add_argument('--exclude', type=str, nargs='+', metavar='plot', default=[],
                       help='Plots to exclude (overrides --plots)')
    g_sel.add_argument('--list-plots', action='store_true', help='List all available plot keys and exit')

    g_fig = p.add_argument_group('Figure & styling')
    g_fig.add_argument('--figwidth', type=float, default=22.0, help='Figure width (in)')
    g_fig.add_argument('--per-plot-height', type=float, default=3.0, help='Height per subplot (in)')
    g_fig.add_argument('--dpi', type=int, default=300, help='DPI')
    g_fig.add_argument('--alpha', type=float, default=0.5, help='Point alpha')
    g_fig.add_argument('--color', type=str, default='purple', help='Default color')
    g_fig.add_argument('--title', type=str, default='Candidates Overview', help='Figure title')

    g_knobs = p.add_argument_group('Plot-specific knobs')
    g_knobs.add_argument('--snr-scale', type=float, default=1.0, help='Bubble size scale for dm_vs_period_bubble')
    g_knobs.add_argument('--period-hist-bins', type=int, default=2000, help='Bins for period histogram')
    g_knobs.add_argument('--dm-hist-bins', type=int, default=20, help='Bins for DM histogram')
    g_knobs.add_argument('--period-xlim', type=float, nargs=2, default=None, metavar=('XMIN','XMAX'),
                         help='xlim for period hist/axes')
    g_knobs.add_argument('--period-vline', type=float, default=None, help='Vertical line in period vs snr plot')
    g_knobs.add_argument('--snr-hist-bin', type=int, default=2000, help='Bin size (width) for SNR histogram')
    g_knobs.add_argument('--snr-hist-xlim', type=float, nargs=2, default=None, metavar=('XMIN','XMAX'),
                         help='X limits for SNR histogram')
    g_knobs.add_argument('--log-period', action='store_true', help='Use log10(period) for P–DM/KDE plots')
    g_knobs.add_argument('--gridsize', type=int, default=60, help='Gridsize for hexbin plots')
    g_knobs.add_argument('--size-scale', type=float, default=2.0, help='Marker size scale for pacc_scatter')
    g_knobs.add_argument('--topk', type=int, default=10, help='Top-K peaks for harmonic overlay')
    g_knobs.add_argument('--harmonic-max', type=int, default=8, help='Max harmonic number for overlays/fundamentalization')
    g_knobs.add_argument('--harmonic-tol', type=float, default=1e-3, help='Relative tolerance for harmonic matching')
    g_knobs.add_argument('--dm-bins', type=float, nargs='+', default=[0,20,100,300,1000],
                         help='Edges for DM binning (facets/ridgeline)')
    g_knobs.add_argument('--nbeams', type=int, default=None, help='Total beams (for nassoc guideline)')
    g_knobs.add_argument('--snr-thresh', type=float, default=None, help='SNR threshold marker')

    g_rfi = p.add_argument_group('RFI landmarks')
    g_rfi.add_argument('--rfi-periods', type=float, nargs='*', default=[1.0, 0.02, 0.0166667],
                       help='Known RFI/clock periods (s) to mark')
    g_rfi.add_argument('--rfi-dm-bands', type=float, nargs=2, default=[0,2], metavar=('DMLO','DMHI'),
                       help='DM band to shade (e.g., 0–2)')

    g_out = p.add_argument_group('Output')
    g_out.add_argument('--outfile', type=str, default='xml_statistics.png',
                       help='Output image filename (combined)')
    g_out.add_argument('--separate', action='store_true',
                       help='Save each selected plot as a separate figure')
    g_out.add_argument('--outdir', type=str, default='figs',
                       help='Directory for --separate outputs')
    g_out.add_argument('--prefix', type=str, default='xmlviz',
                       help='Filename prefix for --separate outputs')

    return p

# -----------------------------
# Main
# -----------------------------

def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.list_plots:
        print("Available plot keys:\n  " + "\n  ".join(sorted(k for k in PLOTTERS)))
        return

    if not args.xml_file:
        parser.error("No XML files provided. Pass paths/globs or use --list-plots.")

    df = load_many_xml(args.xml_file)

    # Filters
    if args.period_max is not None:
        df = df[df['period'] < args.period_max]
    if args.min_snr is not None:
        df = df[df['snr'] >= args.min_snr]

    # Validate plot keys
    invalid = [p for p in args.plots if p not in PLOTTERS]
    if invalid:
        parser.error(f"Unknown plot key(s): {invalid}. Use --list-plots to see options.")

    plots = [p for p in args.plots if p in PLOTTERS]
    if args.exclude:
        plots = [p for p in plots if p not in set(args.exclude)]
    if not plots:
        raise SystemExit('No plots selected after applying --exclude.')

    # ---------- Separate output mode ----------
    if args.separate:
        outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
        for key in plots:
            # Size per-plot; facet gets taller
            facet_rows = None
            if key == 'facet_p_snr_by_dm':
                n = len(args.dm_bins) - 1
                facet_rows = int(np.ceil(n / 2))
                fig = plt.figure(figsize=(args.figwidth, args.per_plot_height * max(1, facet_rows)), dpi=args.dpi,
                                 constrained_layout=True)
                gs = fig.add_gridspec(1, 1)
                slot = gs[0, 0]
                _rows, _axes = plot_facet_p_snr_by_dm_into_slot(fig, slot, df, dm_edges=args.dm_bins)
            else:
                fig = plt.figure(figsize=(args.figwidth, args.per_plot_height), dpi=args.dpi,
                                 constrained_layout=True)
                ax = fig.add_subplot(111)
                # draw
                if key == 'period_vs_snr':
                    plot_period_vs_snr(ax, df, color=args.color, alpha=args.alpha, period_vline=args.period_vline)
                    if args.period_xlim is not None: ax.set_xlim(*args.period_xlim)
                elif key == 'dm_vs_snr':
                    plot_dm_vs_snr(ax, df, color=args.color, alpha=args.alpha)
                elif key == 'acc_vs_snr':
                    plot_acc_vs_snr(ax, df, color=args.color, alpha=args.alpha)
                elif key == 'nassoc_vs_snr':
                    plot_nassoc_snr(ax, df, nbeams=args.nbeams, snr_thresh=args.snr_thresh)
                elif key == 'nh_vs_snr':
                    plot_nh_vs_snr(ax, df, color=args.color, alpha=args.alpha)
                elif key == 'dm_vs_period_bubble':
                    plot_dm_vs_period_bubble(ax, df, color=args.color, alpha=args.alpha, snr_scale=args.snr_scale)
                elif key == 'hist_period':
                    plot_hist_period(ax, df, color=args.color, alpha=args.alpha,
                                     bins=args.period_hist_bins, period_xlim=args.period_xlim)
                elif key == 'hist_dm':
                    plot_hist_dm(ax, df, color=args.color, alpha=args.alpha, bins=args.dm_hist_bins)
                elif key == 'hist_snr':
                    plot_snr_hist(ax, df, bin_size=args.snr_hist_bin, xlims=args.snr_hist_xlim)
                elif key == 'pdm_hex':
                    plot_pdm_hex(ax, df, log_period=args.log_period, gridsize=args.gridsize,
                                 rfi_periods=args.rfi_periods, rfi_dm_bands=args.rfi_dm_bands)
                elif key == 'pacc_scatter':
                    plot_pacc_scatter(ax, df, size_scale=args.size_scale, alpha=args.alpha)
                elif key == 'dmacc_hex':
                    plot_dmacc_hex(ax, df, gridsize=args.gridsize)
                elif key == 'harmonic_overlay':
                    plot_harmonic_overlay(ax, df, topk=args.topk, max_harm=args.harmonic_max, alpha=args.alpha)
                elif key == 'fundamental_hist':
                    plot_fundamental_hist(ax, df, max_harm=args.harmonic_max, bins=200)
                elif key == 'snr_ecdf':
                    plot_snr_ecdf(ax, df, snr_thresh=args.snr_thresh)
                elif key == 'kde_pdm':
                    plot_kde_pdm(ax, df, log_period=args.log_period)

            if args.title:
                fig.suptitle(args.title)
            outfile = outdir / f"{args.prefix}_{key}.png"
            fig.savefig(outfile, dpi=args.dpi)
            plt.close(fig)
            print(f"Saved: {outfile}")
        return

    # ---------- Combined figure (with proper layout) ----------
    # Compute per-plot height ratios (facet needs multiple rows)
    height_ratios = []
    facet_slot_indices = []
    for key in plots:
        if key == 'facet_p_snr_by_dm':
            n = len(args.dm_bins) - 1
            rows = int(np.ceil(n / 2))
            height_ratios.append(rows)  # rows * per_plot_height
            facet_slot_indices.append(len(height_ratios)-1)
        else:
            height_ratios.append(1)

    total_height = args.per_plot_height * sum(height_ratios)
    fig = plt.figure(figsize=(args.figwidth, max(3.0, total_height)), dpi=args.dpi, constrained_layout=True)
    gs = fig.add_gridspec(len(plots), 1, height_ratios=height_ratios)

    for i, key in enumerate(plots):
        if key == 'facet_p_snr_by_dm':
            slot = gs[i, 0]
            plot_facet_p_snr_by_dm_into_slot(fig, slot, df, dm_edges=args.dm_bins)
        else:
            ax = fig.add_subplot(gs[i, 0])
            if key == 'period_vs_snr':
                plot_period_vs_snr(ax, df, color=args.color, alpha=args.alpha, period_vline=args.period_vline)
                if args.period_xlim is not None: ax.set_xlim(*args.period_xlim)
            elif key == 'dm_vs_snr':
                plot_dm_vs_snr(ax, df, color=args.color, alpha=args.alpha)
            elif key == 'acc_vs_snr':
                plot_acc_vs_snr(ax, df, color=args.color, alpha=args.alpha)
            elif key == 'nassoc_vs_snr':
                plot_nassoc_snr(ax, df, nbeams=args.nbeams, snr_thresh=args.snr_thresh)
            elif key == 'nh_vs_snr':
                plot_nh_vs_snr(ax, df, color=args.color, alpha=args.alpha)
            elif key == 'dm_vs_period_bubble':
                plot_dm_vs_period_bubble(ax, df, color=args.color, alpha=args.alpha, snr_scale=args.snr_scale)
            elif key == 'hist_period':
                plot_hist_period(ax, df, color=args.color, alpha=args.alpha, bins=args.period_hist_bins,
                                 period_xlim=args.period_xlim)
            elif key == 'hist_dm':
                plot_hist_dm(ax, df, color=args.color, alpha=args.alpha, bins=args.dm_hist_bins)
            elif key == 'hist_snr':
                plot_snr_hist(ax, df, bin_size=args.snr_hist_bin, xlims=args.snr_hist_xlim)
            elif key == 'pdm_hex':
                plot_pdm_hex(ax, df, log_period=args.log_period, gridsize=args.gridsize,
                             rfi_periods=args.rfi_periods, rfi_dm_bands=args.rfi_dm_bands)
            elif key == 'pacc_scatter':
                plot_pacc_scatter(ax, df, size_scale=args.size_scale, alpha=args.alpha)
            elif key == 'dmacc_hex':
                plot_dmacc_hex(ax, df, gridsize=args.gridsize)
            elif key == 'harmonic_overlay':
                plot_harmonic_overlay(ax, df, topk=args.topk, max_harm=args.harmonic_max, alpha=args.alpha)
            elif key == 'fundamental_hist':
                plot_fundamental_hist(ax, df, max_harm=args.harmonic_max, bins=200)
            elif key == 'snr_ecdf':
                plot_snr_ecdf(ax, df, snr_thresh=args.snr_thresh)
            elif key == 'snr_by_dm_ridgeline':
                plot_snr_by_dm_ridgeline(ax, df, dm_edges=args.dm_bins)
            elif key == 'kde_pdm':
                plot_kde_pdm(ax, df, log_period=args.log_period)

    if args.title:
        fig.suptitle(args.title)

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outfile, dpi=args.dpi)
    print(f"Saved figure to: {Path(args.outfile).resolve()}")

if __name__ == '__main__':
    main()
