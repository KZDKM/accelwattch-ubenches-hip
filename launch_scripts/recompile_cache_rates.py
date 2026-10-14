#!/usr/bin/env python3
"""Rebuild per-benchmark cache hit rates from EXISTING rocprof-compute workloads.

No profiling: only `rocprof-compute analyze`, which reads the saved counters.

Two fixes over the rates written by profile_cache.sh:
  1. Only the benchmark kernel is analyzed. Some binaries also launch helper
     kernels (__amd_rocclr_copyBuffer, convertFp32ToFp16, *pointers_init), and
     analyze otherwise averages "per kernel" metrics across all of them.
  2. Loads and stores get separate rates.

Output columns (fractions in [0, 1]):
  l1_hit        vL1D hit rate, all requests            (16.1.0)
  l2_hit        L2 hit rate, all requests              (17.1.2)
  load_l1_hit   share of vL1D read accesses that hit:
                    1 - L1-L2 Read / (Cache Accesses - L1-L2 Write)
                vL1D is write-through, so every write access is forwarded and
                Cache Accesses - Hits == L1-L2 Read + L1-L2 Write.
  load_l2_hit   share of L2 read requests not sent to HBM:
                    1 - HBM Read / L2 Read Req          (17.6.4 / 17.3.2)
  store_l2_hit  share of L2 write requests not sent to HBM:
                    1 - HBM Write and Atomic / L2 Write Req   (17.6.9 / 17.3.3)

A rate is empty when its denominator is zero (e.g. no stores in the kernel).

Usage:
  ./recompile_cache_rates.py [WORK_DIR] [OUT_CSV]
  defaults: ./cache_workloads  ./cache_workloads/cache_hit_rates_split.csv
"""
import csv
import pathlib
import re
import subprocess
import sys

import pandas as pd

work = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "cache_workloads")
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else work / "cache_hit_rates_split.csv")

# Helper kernels that are not the benchmark itself.
AUX_KERNEL = re.compile(r"^(__amd_rocclr_|convertFp32ToFp16|.*pointers_init)")

SEP = r"[|│]"
ROW = re.compile(rf"^\s*{SEP}\s*([\d.]+)\s*{SEP}\s*([^|│]+?)\s*{SEP}\s*([-\d.eE+]*)\s*{SEP}")


def analyze(workload, dispatch_ids):
    cmd = ["rocprof-compute", "analyze", "-p", str(workload), "--block", "16", "17",
           "-d", *dispatch_ids]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    vals = {}
    for line in res.stdout.splitlines():
        m = ROW.match(line)
        if m and m.group(3):
            try:
                vals[m.group(1)] = float(m.group(3))
            except ValueError:
                pass
    return vals


def ratio_complement(num, den):
    return 1.0 - num / den if den and den > 0 else None


rows, problems = [], []
workloads = sorted(p.parent for p in work.glob("*/pmc_perf.csv"))
for i, wl in enumerate(workloads, 1):
    name = wl.name
    if not (work / f"{name}.done").exists():
        problems.append((name, "no .done marker (incomplete profile), skipped"))
        continue
    disp = pd.read_csv(wl / "pmc_perf.csv", usecols=["Dispatch_ID", "Kernel_Name"])
    keep = disp[~disp.Kernel_Name.str.match(AUX_KERNEL)]
    if keep.empty:
        problems.append((name, "only helper kernels found, skipped"))
        continue
    ids = [str(int(d)) for d in keep.Dispatch_ID]
    v = analyze(wl, ids)
    g = lambda k: v.get(k)
    if g("16.1.0") is None or g("17.1.2") is None:
        problems.append((name, "analyze produced no 16.1/17.1 metrics"))
        continue
    rd_acc = (g("16.3.6") or 0) - (g("16.3.11") or 0)
    r = {
        "benchmark": name,
        "l1_hit": g("16.1.0") / 100.0,
        "l2_hit": g("17.1.2") / 100.0,
        "load_l1_hit": ratio_complement(g("16.3.10") or 0, rd_acc),
        "load_l2_hit": ratio_complement(g("17.6.4") or 0, g("17.3.2")),
        "store_l2_hit": ratio_complement(g("17.6.9") or 0, g("17.3.3")),
        "kernel": keep.Kernel_Name.iloc[0].split("(")[0],
        "helper_dispatches_excluded": len(disp) - len(keep),
    }
    for k in ("l1_hit", "l2_hit", "load_l1_hit", "load_l2_hit", "store_l2_hit"):
        if r[k] is not None and not -0.02 <= r[k] <= 1.02:
            problems.append((name, f"{k} = {r[k]:.3f} outside [0,1]"))
        if r[k] is not None:
            r[k] = min(max(r[k], 0.0), 1.0)
    rows.append(r)
    print(f"[{i}/{len(workloads)}] {name}", flush=True)

cols = ["benchmark", "l1_hit", "l2_hit", "load_l1_hit", "load_l2_hit", "store_l2_hit",
        "kernel", "helper_dispatches_excluded"]
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else ("" if r[k] is None else r[k]))
                    for k in cols})
print(f"\nwrote {out}  ({len(rows)} workloads)")
for name, msg in problems:
    print(f"WARNING: {name}: {msg}", file=sys.stderr)
