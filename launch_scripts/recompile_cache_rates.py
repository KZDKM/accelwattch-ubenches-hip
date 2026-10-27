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

Raw per-kernel counts (benchmark kernel only, analyze's per-kernel normalisation),
for consumers that need absolute traffic rather than ratios:
  l1_l2_reads   vL1D -> L2 read requests         (16.3.10)
  l2_read_req   L2 read requests                 (17.3.2)
  l2_write_req  L2 write requests                (17.3.3)
  hbm_reads     L2 -> HBM read requests          (17.6.4)
  hbm_writes    L2 -> HBM write/atomic requests  (17.6.9)
  l1_l2_writes  vL1D -> L2 write requests        (16.3.11)
  hbm_rd_32b / hbm_rd_64b / hbm_rd_128b   HBM read requests by size  (17.6.0-2)
  hbm_wr_32b / hbm_wr_64b                 HBM write requests by size (17.6.6, 17.6.8)
  hbm_read_bytes   32*rd_32b + 64*rd_64b + 128*rd_128b
  hbm_write_bytes  32*wr_32b + 64*wr_64b

A rate is empty when its denominator is zero (e.g. no stores in the kernel) or
when it falls outside [0, 1] (a counter inconsistency; reported as a warning).

A workload counts as complete when analyze returns every metric used below;
.done markers are not required.

Incremental by default: rows in an existing OUT_CSV are reused for workloads
whose pmc_perf.csv is older than that file. Pass --full to redo everything.
--only REGEX re-analyzes just the workloads whose name matches REGEX
(re.match) and copies every other row from OUT_CSV unchanged.

Usage:
  ./recompile_cache_rates.py [--full | --only REGEX] [WORK_DIR] [OUT_CSV]
  defaults: ./cache_workloads  ./cache_workloads/cache_hit_rates_split.csv
"""
import csv
import pathlib
import re
import subprocess
import sys

import pandas as pd

argv = sys.argv[1:]
only = None
if "--only" in argv:
    i = argv.index("--only")
    only = re.compile(argv[i + 1])
    del argv[i:i + 2]
args = [a for a in argv if a != "--full"]
full = "--full" in argv
work = pathlib.Path(args[0] if len(args) > 0 else "cache_workloads")
out = pathlib.Path(args[1] if len(args) > 1 else work / "cache_hit_rates_split.csv")
REQUIRED = ["16.1.0", "17.1.2", "16.3.6", "16.3.10", "16.3.11", "17.3.2", "17.3.3", "17.6.4", "17.6.9"]

previous = {}
if out.exists() and not full:
    prev_mtime = out.stat().st_mtime
    with out.open() as f:
        for r in csv.DictReader(f):
            previous[r["benchmark"]] = (r, prev_mtime)

# Helper kernels that are not the benchmark itself.
AUX_KERNEL = re.compile(r"^(__amd_rocclr_|convertFp32ToFp16|.*pointers_init|fill_bytes)")

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
    if only is not None and not only.match(name):
        if name in previous:
            rows.append(previous[name][0])
        continue
    if only is None and name in previous and (wl / "pmc_perf.csv").stat().st_mtime < previous[name][1]:
        rows.append(previous[name][0])
        continue
    disp = pd.read_csv(wl / "pmc_perf.csv", usecols=["Dispatch_ID", "Kernel_Name"])
    keep = disp[~disp.Kernel_Name.str.match(AUX_KERNEL)]
    if keep.empty:
        problems.append((name, "only helper kernels found, skipped"))
        continue
    ids = [str(int(d)) for d in keep.Dispatch_ID]
    v = analyze(wl, ids)
    g = lambda k: v.get(k)
    missing = [k for k in REQUIRED if g(k) is None]
    if missing:
        problems.append((name, f"incomplete or no cache traffic: missing metrics {missing}"))
        continue
    rd_acc = (g("16.3.6") or 0) - (g("16.3.11") or 0)
    r = {
        "benchmark": name,
        "l1_hit": g("16.1.0") / 100.0,
        "l2_hit": g("17.1.2") / 100.0,
        "load_l1_hit": ratio_complement(g("16.3.10") or 0, rd_acc),
        "load_l2_hit": ratio_complement(g("17.6.4") or 0, g("17.3.2")),
        "store_l2_hit": ratio_complement(g("17.6.9") or 0, g("17.3.3")),
        "l1_l2_reads": g("16.3.10"),
        "l2_read_req": g("17.3.2"),
        "l2_write_req": g("17.3.3"),
        "hbm_reads": g("17.6.4"),
        "hbm_writes": g("17.6.9"),
        "l1_l2_writes": g("16.3.11"),
        "hbm_rd_32b": g("17.6.0"),
        "hbm_rd_64b": g("17.6.1"),
        "hbm_rd_128b": g("17.6.2"),
        "hbm_wr_32b": g("17.6.6"),
        "hbm_wr_64b": g("17.6.8"),
        "kernel": keep.Kernel_Name.iloc[0].split("(")[0],
        "helper_dispatches_excluded": len(disp) - len(keep),
    }
    sizes = [g(k) for k in ("17.6.0", "17.6.1", "17.6.2")]
    r["hbm_read_bytes"] = (32 * sizes[0] + 64 * sizes[1] + 128 * sizes[2]
                           if None not in sizes else None)
    wsizes = [g(k) for k in ("17.6.6", "17.6.8")]
    r["hbm_write_bytes"] = (32 * wsizes[0] + 64 * wsizes[1] if None not in wsizes else None)
    for k in ("l1_hit", "l2_hit", "load_l1_hit", "load_l2_hit", "store_l2_hit"):
        if r[k] is not None and not -0.02 <= r[k] <= 1.02:
            problems.append((name, f"{k} = {r[k]:.3f} outside [0,1], left blank"))
            r[k] = None
        elif r[k] is not None:
            r[k] = min(max(r[k], 0.0), 1.0)
    rows.append(r)
    print(f"[{i}/{len(workloads)}] {name}", flush=True)

cols = ["benchmark", "l1_hit", "l2_hit", "load_l1_hit", "load_l2_hit", "store_l2_hit",
        "l1_l2_reads", "l2_read_req", "l2_write_req", "hbm_reads", "hbm_writes",
        "l1_l2_writes", "hbm_rd_32b", "hbm_rd_64b", "hbm_rd_128b", "hbm_wr_32b", "hbm_wr_64b",
        "hbm_read_bytes", "hbm_write_bytes",
        "kernel", "helper_dispatches_excluded"]
RAW = {"l1_l2_reads", "l2_read_req", "l2_write_req", "hbm_reads", "hbm_writes",
       "l1_l2_writes", "hbm_rd_32b", "hbm_rd_64b", "hbm_rd_128b", "hbm_wr_32b", "hbm_wr_64b",
       "hbm_read_bytes", "hbm_write_bytes"}


def fmt(k, v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.0f}" if k in RAW else f"{v:.6f}"
    return v                     # reused rows are already strings


with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: fmt(k, r.get(k)) for k in cols})
reused = sum(1 for r in rows if isinstance(r.get("l1_hit"), str))
print(f"\nwrote {out}  ({len(rows)} workloads, {reused} reused, {len(rows) - reused} analyzed)")
for name, msg in problems:
    print(f"WARNING: {name}: {msg}", file=sys.stderr)
