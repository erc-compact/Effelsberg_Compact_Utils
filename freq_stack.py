#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from sigpyproc.readers import FilReader

def freq_stack_debug(infiles, outfile, block_size=1024):
    """
    Stack multiple .fil files in frequency, time-aligned.
    
    Author: Fazal Kareem
    Date: 18.05.2025
    """
    # 1) Open all inputs
    readers = [FilReader(f) for f in infiles]
    print(f"[DEBUG] Opened {len(readers)} files")

    # 2) Compute per-file start MJD and end MJD
    headers = [r.header for r in readers]
    tstarts = [h.tstart for h in headers]
    tsamps  = [h.tsamp for h in headers]  # µs→s
    print(tsamps)
    ends    = [t0 + (h.nsamples - 1)*dt/86400.0
               for h, t0, dt in zip(headers, tstarts, tsamps)]
    mjd_start = max(tstarts)
    mjd_end   = min(ends)
    print("[DEBUG] tstarts = " + ", ".join(f"{t:.15f}" for t in tstarts))
    print("[DEBUG] ends    = "   + ", ".join(f"{e:.15f}" for e in ends))
    print(f"[DEBUG] Align window: {mjd_start:.15f} → {mjd_end:.15f}")
    if mjd_end <= mjd_start:
        sys.exit("ERROR: No overlapping time window!")

    # 3) Compute sample skips & common length
    skips = [int(round((mjd_start - t0)*86400.0 / dt))
             for t0, dt in zip(tstarts, tsamps)]
    avails = [h.nsamples - sk for h, sk in zip(headers, skips)]
    total = min(avails)
    print(f"[DEBUG] skips  = {skips}")
    print(f"[DEBUG] avails = {avails}")
    print(f"[DEBUG] total  = {total} samples")
    if total <= 0:
        sys.exit("ERROR: No samples to write after alignment!")

    # 4) Build global frequency axis
    foff        = headers[0].foff
    all_freqs   = np.hstack([h.chan_freqs for h in headers])
    fmin, fmax  = all_freqs.min(), all_freqs.max()
    nchan_total = int(abs(round((fmax - fmin)/abs(foff)))) + 1
    print(f"[DEBUG] Global band: {fmin:.4f}–{fmax:.4f} MHz, {nchan_total} chans")

    # 5) Prepare output header & writer
    base_hdr  = headers[0]
    new_fch1  = fmin if foff>0 else fmax
    new_hdr   = base_hdr.new_header({
        "nchans":   nchan_total,
        "foff":     foff,
        "fch1":     new_fch1,
        "nsamples": total
    })
    if os.path.exists(outfile):
        os.remove(outfile)
    writer = new_hdr.prep_outfile(outfile)
    print(f"[DEBUG] Writing to: {outfile}")

    # 6) Compute per-file channel offsets
    global_freqs = new_hdr.chan_freqs
    print(f"[DEBUG] Global freqs : {global_freqs}")
    chan_offsets = [
        int(np.argmin(np.abs(global_freqs - h.fch1)))
        for h in headers
    ]
    print(f"[DEBUG] Channel offsets: {chan_offsets}")

    # 7) Stream & merge 
    written = 0
    iter_no = 0
    while written < total:
        nread = min(block_size, total - written)
        print(f"[DEBUG] ITER {iter_no}: start_sample={written}, nread={nread}")

        # allocate time × freq array
        merged = np.zeros((nread, nchan_total), dtype=np.uint8)

        for i, (r, sk, offs) in enumerate(zip(readers, skips, chan_offsets)):
            blk = r.read_block(sk + written, nread)
            arr = blk.data.T
            print(f"    [DEBUG] file#{i} read_block → shape {arr.shape}, sum={arr.sum():.0f}")
            merged[:, offs:offs + arr.shape[1]] = arr

        # write out
        writer.cwrite(merged.flatten("C"))
        written += nread
        iter_no += 1

    # 8) Finalize
    writer.close()
    size = os.path.getsize(outfile)
    print(f"[DEBUG] Done: iterations={iter_no}, wrote={written} samples, file_size={size:,} bytes")
    return outfile

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Frequency-stack with debug.")
    p.add_argument("infiles", nargs="+", help="Input .fil files")
    p.add_argument("-o","--output", required=True, help="Output stacked .fil")
    p.add_argument("-B","--block-size", type=int, default=1024,
                   help="Samples per block (default=1024)")
    args = p.parse_args()
    
    out = freq_stack_debug(args.infiles, args.output, block_size=args.block_size)
    if not os.path.exists(out):
        sys.exit(f"ERROR: {out} not created")
    print(f"SUCCESS: wrote {out}")
