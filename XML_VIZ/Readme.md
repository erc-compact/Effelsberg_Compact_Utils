# XML Candidate Visualizer (pandas)

A flexible tool to explore pulsar-search candidate XML files and generate rich diagnostic plots. It’s designed for fast triage of real pulsars vs RFI using just the fields you already have (period, DM, SNR, acc, nassoc, nh).

## Quick start

```bash
# Everything at once (default bundle)
viz_xml_candidates.py *.xml --outfile all_plots.png

# List all plot keys
viz_xml_candidates.py *.xml --list-plots

# Pick and order plots
viz_xml_candidates.py *.xml \
  --plots pdm_hex kde_pdm harmonic_overlay fundamental_hist \
  --log-period --rfi-periods 1.0 0.02 0.0166667 --rfi-dm-bands 0 2 \
  --outfile discovery_suite.png
```

---

## Inputs & filters

* **xml\_file**: paths or globs to one or more candidate XML files.
* **--period-max**: keep only candidates with period < value (default: 0.01 s).
* **--min-snr**: drop faint candidates below SNR.

## Plot selection

* **--plots**: list of plot keys to render, in order. Use `--list-plots` to see them.
* **--exclude**: remove some plots from the selected list.

## Figure & styling

* **--figwidth**, **--per-plot-height**, **--dpi**: size/quality.
* **--title**: suptitle for the combined figure.

## RFI landmarks (optional)

* **--rfi-periods**: vertical period lines (e.g., 1 s, 50/60 Hz harmonics: 0.02 / 0.01667 s).
* **--rfi-dm-bands**: shaded DM band (e.g., 0–2 pc/cm³).

---

## Plot reference & use cases

### 1) `pdm_hex` — Period–DM hexbin (SNR-weighted)

**Why**: Real pulsars cluster at non-zero DM; RFI stacks at DM≈0. Hexbin avoids overplotting.
**Knobs**: `--log-period`, `--gridsize`, `--rfi-periods`, `--rfi-dm-bands`.

### 2) `kde_pdm` — KDE contours on Period–DM

**Why**: Smooth density estimate to reveal islands.
**Note**: Uses SciPy if available; falls back to blurred 2D hist.

### 3) `harmonic_overlay` — Harmonics from top‑K

**Why**: Quickly identify harmonic families from strong peaks.
**Knobs**: `--topk`, `--harmonic-max`.

### 4) `fundamental_hist` — Fundamentalized period histogram

**Why**: Collapses harmonics to emphasize unique periods.
**Knobs**: `--harmonic-max`, internal tolerance is conservative.

### 5) `pacc_scatter` — Period vs Acceleration (color/size = SNR)

**Why**: Surfacing accelerated binary candidates.
**Knobs**: `--size-scale` for marker size.

### 6) `dmacc_hex` — DM vs Acceleration (SNR-weighted hexbin)

**Why**: Acceleration vs dispersion regimes; highlights binary-rich zones.

### 7) `nassoc_snr` — nassoc vs SNR (color=DM)

**Why**: Multi-beam RFI often has high `nassoc` at DM≈0. Add threshold lines.
**Knobs**: `--nbeams`, `--snr-thresh`.

### 8) `snr_ecdf` — Empirical CDF of SNR

**Why**: Choose a principled SNR cut; compare runs.
**Knobs**: `--snr-thresh`.

### 9) `facet_p_snr_by_dm` — Small multiples of Period–SNR by DM bands

**Why**: Reduces occlusion; see how P‐SNR varies by DM range.
**Knobs**: `--dm-bins` (edges).

### 10) `snr_by_dm_ridgeline` — SNR distribution by DM bins

**Why**: Where is the SNR concentrated? Peaks indicate promising DM bands.
**Knobs**: `--dm-bins`.

### 11) `hist_snr` — SNR histogram

**Why**: Simple distribution view; sanity-check tails.
**Knobs**: `--snr-hist-bin` (bin width; default 2000), `--snr-hist-xlim`.

### 12) Legacy/utility plots

* `period_vs_snr`, `dm_vs_snr`, `acc_vs_snr`, `dm_vs_period_bubble`, `hist_period`, `hist_dm`, `nassoc_vs_snr`, `nh_vs_snr`.

---

## Tips

* Use `--log-period` when you have a very wide period range.
* For crowded fields, prefer `pdm_hex` + `kde_pdm` over raw scatter.
* Set `--dm-bins` to match survey heuristics (e.g., 0–20, 20–100, 100–300, 300+).
* Add `--snr-thresh` to align ECDF/hist guides to your pipeline cut.

## Troubleshooting

* “No XML files found”: check your shell glob quoting.
* Slow rendering with huge candidate sets: try bigger `--gridsize` and fewer plots per run.
* KDE contours missing: install SciPy (`pip install scipy`) or rely on fallback.

## License

MIT (feel free to adapt for your pipeline).

