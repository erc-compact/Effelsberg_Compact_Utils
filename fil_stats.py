#!/usr/bin/env python3

from sigpyproc.readers import FilReader
import numpy as np
import matplotlib.pyplot as plt
import argparse

# === Load filterbank ===

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help='Path to filterbank file')
parser.add_argument('-o', type=str, help='Output name', default='channel_stats_all.png')
parser.add_argument('-g', type=int, help='Gulp size', default=100000)
parser.add_argument('-n', type=int, help='Number of samples to plot', default=None)
args = parser.parse_args()
fil = FilReader(f"{args.f}")
if args.n == None:
    nsamples_total = fil.header.nsamples
else:
    nsamples_total = args.n

gulp_size = args.g

nchans = fil.header.nchans


# Per-gulp channelwise arrays
mean_list = []
std_list = []

# Per-gulp median of channelwise stats
median_means = []
median_stds = []

for i in range(0, nsamples_total, gulp_size):
    nsamps = min(gulp_size, nsamples_total - i)
    block = fil.read_block(i, nsamps)

    data = block.data.astype(np.float64).reshape((nchans, nsamps), order='F')

    # Channelwise stats
    mean_ch = data.mean(axis=1)
    std_ch = data.std(axis=1)

    # Median of channelwise stats
    median_mean = np.median(mean_ch)
    median_std = np.median(std_ch)

    mean_list.append(mean_ch)
    std_list.append(std_ch)
    median_means.append(median_mean)
    median_stds.append(median_std)

    print(f"Gulp {i}: Median Mean = {median_mean:.2f}, Median Std = {median_std:.2f}, Ch0 Mean = {mean_ch[0]:.2f}")

# === Stack everything ===
mean_arr = np.stack(mean_list)        # shape (n_gulps, n_chans)
std_arr = np.stack(std_list)
median_means = np.array(median_means) # shape (n_gulps,)
median_stds = np.array(median_stds)

gulp_indices = np.arange(len(median_means))

# === Save data to compressed .npz file ===
np.savez("filterbank_stats.npz",
         mean_arr=mean_arr,
         std_arr=std_arr,
         median_means=median_means,
         median_stds=median_stds)
print("Saved stats to filterbank_stats.npz")

# === Plot ===
plt.figure(figsize=(14, 8))

# --- Median of channelwise means/stds ---
plt.subplot(3, 1, 1)
plt.plot(gulp_indices, median_means, label="Median of Channel Means")
plt.plot(gulp_indices, median_stds, label="Median of Channel Stds")
plt.ylabel("Value")
plt.title(f"Filterbank Statistics {args.f}")
plt.legend()
plt.grid(True)

# --- Channelwise mean heatmap ---
plt.subplot(3, 1, 2)
plt.imshow(mean_arr.T, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label="Mean")
plt.ylabel("Channel")
plt.title("Channel-wise Mean per Gulp")

# --- Channelwise std heatmap ---
plt.subplot(3, 1, 3)
plt.imshow(std_arr.T, aspect='auto', cmap='plasma', origin='lower')
plt.colorbar(label="Std Dev")
plt.xlabel("Gulp Index")
plt.ylabel("Channel")
plt.title("Channel-wise Std Dev per Gulp")

plt.tight_layout()
plt.savefig(args.o)
