#!/usr/bin/env python3
"""Integrate power CSVs into per-run energies and aggregate them per benchmark.

Reads every power CSV in a directory, integrates power over time to get total
and dynamic energy, optionally windows each trace to the kernel interval, and
writes:

    runs.csv       one row per (benchmark, run) -- inspect this first
    aggregate.csv  one row per benchmark, for the energy-table notebook

Nothing here is trusted blindly: --inspect prints the columns and a few rows so
the column auto-detection can be checked before a full run, and every run row
carries the numbers needed to sanity-check the integral by hand.

Usage:
    ./integrate_power.py --inspect
    ./integrate_power.py --static 310 --log batch_stdout.txt
    ./integrate_power.py --static 310 --kernel-times kernels.csv --iterations 1000000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------- file naming

# profiling_result_BAR_profiled_run1.csv_0.csv, BAR_profiled_run1.csv, BAR_run3.csv
NAME_PATTERNS = [
    re.compile(r"^profiling_result_(?P<bench>.+?)_profiled(?:_run(?P<run>\d+))?\.csv"),
    re.compile(r"^(?P<bench>.+?)_profiled(?:_run(?P<run>\d+))?\.csv"),
    re.compile(r"^(?P<bench>.+?)_run(?P<run>\d+)\.csv"),
]

# "Running BSSY (run 1/5)" then "gpu execution time = 6054.027 ms"
LOG_RUNNING = re.compile(r"Running\s+(?P<bench>\S+)\s+\(run\s+(?P<run>\d+)")
LOG_KERNEL_MS = re.compile(r"gpu execution time\s*=\s*(?P<ms>[\d.]+)\s*ms")
LOG_KERNEL_S = re.compile(r"gpu execution time\s*=\s*(?P<s>[\d.]+)\s*s\b")

# ------------------------------------------------------------ column guessing

TIME_HINTS = ("timestamp", "time_ns", "time_us", "time_ms", "time", "ts", "elapsed", "secs")
POWER_HINTS = ("power", "watt", "pwr")
# Preferred power columns, most specific first.
POWER_PREFER = ("socket", "average", "avg", "current", "instant")


def parse_name(path: Path) -> tuple[str, int | None]:
    for pattern in NAME_PATTERNS:
        match = pattern.match(path.name)
        if match:
            run = match.group("run") if "run" in match.groupdict() else None
            return match.group("bench"), int(run) if run else None
    return path.stem, None


def read_csv(path: Path, skip: int = 1) -> tuple[list[str], list[dict], list[str]]:
    """Return (header, rows, preamble). The first `skip` lines are parameter junk."""
    with path.open(newline="", errors="replace") as handle:
        preamble = [handle.readline().rstrip("\r\n") for _ in range(skip)]
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
        return (reader.fieldnames or []), rows, preamble


def numeric_columns(header: list[str], rows: list[dict]) -> set[str]:
    """Columns whose first few non-empty values all parse as floats."""
    numeric = set()
    for name in header:
        seen = 0
        for row in rows:
            value = (row.get(name) or "").strip()
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                break
            seen += 1
            if seen >= 5:
                break
        else:
            if seen:
                numeric.add(name)
        if seen >= 5:
            numeric.add(name)
    return numeric


def pick_column(header: list[str], hints: tuple[str, ...],
                allowed: set[str], prefer: tuple[str, ...] = ()) -> str | None:
    candidates = [c for c in header
                  if c in allowed and any(h in c.lower() for h in hints)]
    if not candidates:
        return None
    for token in prefer:
        for c in candidates:
            if token in c.lower():
                return c
    return candidates[0]


def guess_time_scale(values: list[float], column: str) -> tuple[float, str]:
    """Return (seconds per unit, why). Checked against the median sample gap."""
    lowered = column.lower()
    for suffix, scale, unit in (("_ns", 1e-9, "ns"), ("_us", 1e-6, "us"),
                                ("_ms", 1e-3, "ms"), ("_s", 1.0, "s")):
        if lowered.endswith(suffix):
            return scale, f"from column name ({unit})"

    diffs = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not diffs:
        return 1.0, "no positive gaps; assuming seconds"
    gap = statistics.median(diffs)
    # Assume real sampling gaps are between 1 ms and 10 s.
    for scale, unit in ((1e-9, "ns"), (1e-6, "us"), (1e-3, "ms"), (1.0, "s")):
        if 1e-3 <= gap * scale <= 10:
            return scale, f"median gap {gap:g} -> {gap * scale:g}s ({unit})"
    return 1.0, f"median gap {gap:g}; fell back to seconds"


# ------------------------------------------------------- clock-based windows

def clock_window(times: list[float], clocks: list[float], threshold: float,
                 min_frac: float, gap: float) -> tuple[tuple[float, float], float] | None:
    """Longest stretch where clock >= threshold, tolerating brief dips.

    Segments separated by less than `gap` seconds are merged: a memory-bound
    kernel can dip under the threshold mid-run, and taking only the longest
    strictly-contiguous segment would keep a sliver of the real interval.

    Returns ((start, end), coverage) where coverage is the fraction of samples
    inside the window that are actually above threshold -- low coverage means
    the merge spanned a lot of below-threshold time and the window is suspect.

    Clocks ramp on queue submission and linger after the kernel retires, so this
    window is typically a little WIDER than the kernel itself.
    """
    segments, current = [], None
    for t, c in zip(times, clocks):
        if c >= threshold:
            current = [t, t] if current is None else [current[0], t]
        elif current:
            segments.append(current)
            current = None
    if current:
        segments.append(current)
    if not segments:
        return None

    merged = [segments[0]]
    for seg in segments[1:]:
        if seg[0] - merged[-1][1] <= gap:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    best = max(merged, key=lambda s: s[1] - s[0])
    span = times[-1] - times[0]
    if span > 0 and (best[1] - best[0]) / span < min_frac:
        return None

    inside = [c for t, c in zip(times, clocks) if best[0] <= t <= best[1]]
    coverage = sum(1 for c in inside if c >= threshold) / len(inside) if inside else 0.0
    return (best[0], best[1]), coverage


def resolve_threshold(spec: str, clocks: list[float]) -> float:
    """Resolve a clock threshold against one trace.

    'mid' / '0.5m'  fraction of the way from the trace's min clock to its max.
                    This separates idle from active, which is what we want: a
                    memory-bound kernel can sustain a clock well below peak, and
                    a peak-relative threshold would exclude the whole kernel.
    '0.9x'          fraction of the trace's MAX clock. Only safe when the kernel
                    actually runs near peak.
    '600'           absolute MHz.
    """
    spec = spec.strip().lower()
    lo, hi = min(clocks), max(clocks)
    if spec in ("mid", "midpoint"):
        return lo + 0.5 * (hi - lo)
    if spec.endswith("m"):
        return lo + float(spec[:-1]) * (hi - lo)
    if spec.endswith("x"):
        return float(spec[:-1]) * hi
    return float(spec)


# ------------------------------------------------------------------ integrate

def integrate(times: list[float], power: list[float], static: float,
              window: tuple[float, float] | None,
              extras: dict[str, list[float]] | None = None) -> dict:
    """Trapezoidal integral of power, and of max(power - static, 0).

    `extras` are extra per-sample series (clock, temperature) summarised over
    exactly the same window as the integral, so they describe the interval the
    energy came from rather than the whole trace.
    """
    extras = extras or {}
    if window:
        lo, hi = window
        keep = [i for i, t in enumerate(times) if lo <= t <= hi]
        if len(keep) >= 2:
            times = [times[i] for i in keep]
            power = [power[i] for i in keep]
            extras = {k: [v[i] for i in keep] for k, v in extras.items()}
        else:
            window = None  # too few samples; fall back to the full trace

    total = dynamic = 0.0
    for (t0, p0), (t1, p1) in zip(zip(times, power), zip(times[1:], power[1:])):
        dt = t1 - t0
        if dt <= 0:
            continue
        total += 0.5 * (p0 + p1) * dt
        d0, d1 = max(p0 - static, 0.0), max(p1 - static, 0.0)
        dynamic += 0.5 * (d0 + d1) * dt

    duration = times[-1] - times[0] if len(times) >= 2 else 0.0
    result = {
        "samples": len(times),
        "duration_s": duration,
        "total_energy_J": total,
        "dynamic_energy_J": dynamic,
        "mean_power_W": total / duration if duration > 0 else 0.0,
        "mean_dynamic_power_W": dynamic / duration if duration > 0 else 0.0,
        "max_power_W": max(power) if power else 0.0,
        "min_power_W": min(power) if power else 0.0,
        "windowed": bool(window),
    }
    for name, series in extras.items():
        if series:
            result[f"{name}_median"] = statistics.median(series)
            result[f"{name}_min"] = min(series)
            result[f"{name}_max"] = max(series)
    return result


# -------------------------------------------------------------- kernel timing

def load_kernel_times(args) -> dict[tuple[str, int | None], float]:
    """(benchmark, run) -> kernel seconds. run None means "applies to all runs"."""
    table: dict[tuple[str, int | None], float] = {}

    if args.log:
        current = None
        for line in Path(args.log).read_text(errors="replace").splitlines():
            running = LOG_RUNNING.search(line)
            if running:
                current = (running.group("bench"), int(running.group("run")))
                continue
            ms = LOG_KERNEL_MS.search(line)
            sec = LOG_KERNEL_S.search(line)
            if (ms or sec) and current:
                table[current] = float(ms.group("ms")) / 1000 if ms else float(sec.group("s"))
                current = None

    if args.kernel_times:
        path = Path(args.kernel_times)
        if path.suffix == ".json":
            for key, value in json.loads(path.read_text()).items():
                table[(key, None)] = float(value)
        else:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    bench = row.get("benchmark") or row.get("bench")
                    run = row.get("run")
                    seconds = row.get("kernel_s")
                    if seconds is None and row.get("kernel_ms") is not None:
                        seconds = float(row["kernel_ms"]) / 1000
                    if bench and seconds is not None:
                        table[(bench, int(run) if run else None)] = float(seconds)

    return table


def lookup_kernel(table, bench: str, run: int | None) -> float | None:
    return table.get((bench, run)) or table.get((bench, None))


# ----------------------------------------------------------------- reporting

def inspect(paths: list[Path], skip: int) -> int:
    for path in paths[:3]:
        header, rows, preamble = read_csv(path, skip)
        print(f"=== {path.name}  ({len(rows)} data rows) ===")
        for i, line in enumerate(preamble, 1):
            print(f"skipped line {i}: {line}")
        print("columns:", header)
        allowed = numeric_columns(header, rows)
        print("numeric:", sorted(allowed))
        print("time guess :", pick_column(header, TIME_HINTS, allowed))
        print("power guess:", pick_column(header, POWER_HINTS, allowed, POWER_PREFER))
        for row in rows[:3]:
            print("  ", {k: row[k] for k in header[:8]})
        print()
    return 0


def aggregate(runs: list[dict], iterations: str | None,
              extra_fields: list[str] | None = None) -> list[dict]:
    by_bench: dict[str, list[dict]] = {}
    for row in runs:
        by_bench.setdefault(row["benchmark"], []).append(row)

    out = []
    for bench in sorted(by_bench):
        group = by_bench[bench]
        dyn = [r["dynamic_energy_J"] for r in group]
        tot = [r["total_energy_J"] for r in group]
        dur = [r["duration_s"] for r in group]
        name = f"{bench}_{iterations}" if iterations else bench
        row = {
            "benchmark": name,
            "count_rows": len(group),
            "calculated_util_energy_dynamic_median": f"{statistics.median(dyn):.6f}",
            "calculated_util_energy_dynamic_mean": f"{statistics.fmean(dyn):.6f}",
            "calculated_util_energy_dynamic_min": f"{min(dyn):.6f}",
            "calculated_util_energy_dynamic_max": f"{max(dyn):.6f}",
            "calculated_util_energy_dynamic_std":
                f"{statistics.stdev(dyn):.6f}" if len(dyn) > 1 else "",
            "measured_energy_median": f"{statistics.median(tot):.6f}",
            "execution_time_ms_median": f"{statistics.median(dur) * 1000:.3f}",
            "mean_dynamic_power_W":
                f"{statistics.median(dyn) / statistics.median(dur):.2f}"
                if statistics.median(dur) > 0 else "0",
            "max_power_W": f"{max(r['max_power_W'] for r in group):.2f}",
            "windowed": "yes" if all(r["windowed"] for r in group) else
                        ("partial" if any(r["windowed"] for r in group) else "no"),
        }
        for base in sorted({f.rsplit("_", 1)[0] for f in (extra_fields or [])}):
            meds = [r[f"{base}_median"] for r in group if f"{base}_median" in r]
            lows = [r[f"{base}_min"] for r in group if f"{base}_min" in r]
            highs = [r[f"{base}_max"] for r in group if f"{base}_max" in r]
            if meds:
                row[f"{base}_median"] = f"{statistics.median(meds):.1f}"
                row[f"{base}_min"] = f"{min(lows):.1f}"
                row[f"{base}_max"] = f"{max(highs):.1f}"
        out.append(row)
    return out


def check(runs: list[dict], static: float) -> None:
    """Flag rows whose numbers cannot be right."""
    problems = []
    for r in runs:
        label = f"{r['benchmark']} run{r['run']}"
        ceiling = max(r["max_power_W"] - static, 0.0)
        if r["mean_dynamic_power_W"] > ceiling + 0.5:
            problems.append(f"{label}: mean dynamic {r['mean_dynamic_power_W']:.1f}W "
                            f"exceeds max-static ceiling {ceiling:.1f}W")
        if r["samples"] < 10:
            problems.append(f"{label}: only {r['samples']} samples")
        if r["duration_s"] <= 0:
            problems.append(f"{label}: non-positive duration")
        ratio = r.get("clock_vs_kernel")
        if ratio is not None and not (0.8 <= ratio <= 1.5):
            problems.append(f"{label}: clock-high window is {ratio:.2f}x the reported "
                            f"kernel time ({r['clock_window_s']:.2f}s vs {r['kernel_s']}s)")
        cov = r.get("clock_coverage")
        if cov is not None and cov < 0.9:
            problems.append(f"{label}: only {cov:.0%} of the window is above the clock "
                            "threshold; the merge may have spanned idle time")
        frac = r.get("window_frac_of_trace")
        if frac is not None and frac < 0.02 and r["duration_s"] < 10:
            problems.append(f"{label}: window is {frac:.1%} of the trace "
                            f"({r['duration_s']:.2f}s) -- kernel may not have run, or "
                            "the clock window fragmented")
        if r["dynamic_energy_J"] == 0:
            problems.append(f"{label}: zero dynamic energy "
                            f"(peak {r['max_power_W']:.0f}W, clock never exceeded "
                            f"{r.get('gfx_clock_MHz_max', 0):.0f} MHz) -- likely a dead kernel")
    if problems:
        print("\nwarnings:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", nargs="?", default=".", type=Path)
    p.add_argument("--glob", default="*.csv")
    p.add_argument("--skip-lines", type=int, default=1,
                   help="preamble lines before the CSV header (default 1)")
    p.add_argument("--static", type=float, default=310.0,
                   help="static power in W subtracted per sample (default 310)")
    p.add_argument("--time-col")
    p.add_argument("--power-col")
    p.add_argument("--time-unit", choices=["s", "ms", "us", "ns"],
                   help="override the timestamp unit instead of guessing")
    p.add_argument("--log", help="batch stdout containing 'gpu execution time = ... ms'")
    p.add_argument("--kernel-times", help="CSV (benchmark,run,kernel_ms|kernel_s) or JSON")
    p.add_argument("--window-mode", choices=["kernel", "clock", "auto"], default="clock",
                   help="how to find the kernel interval: a clock threshold "
                        "(default), the reported kernel time, or kernel time with "
                        "clock as fallback")
    p.add_argument("--clock-col", default="gfx_clock_MHz")
    p.add_argument("--clock-threshold", default="0.5x",
                   help="'0.5x' (default): that fraction of the trace's PEAK clock. "
                        "Half of peak sits well above idle and well below any real "
                        "kernel plateau, including memory-bound kernels that sustain "
                        "far below peak. 'mid'/'0.3m': fraction of the idle-to-peak "
                        "range (fragile when a trace has little idle). '600': absolute MHz.")
    p.add_argument("--clock-gap", type=float, default=2.0,
                   help="merge clock-high segments separated by less than this many "
                        "seconds (default 2.0); prevents mid-kernel dips fragmenting "
                        "the window")
    p.add_argument("--clock-min-frac", type=float, default=0.005,
                   help="reject a clock window shorter than this fraction of the trace")
    p.add_argument("--tail-pad", type=float, default=0.0,
                   help="seconds between kernel end and last sample (default 0)")
    p.add_argument("--extra-cols", default="gfx_clock_MHz,temperature_junction_C",
                   help="comma-separated columns summarised over the same window "
                        "(median/min/max); default clock and junction temperature")
    p.add_argument("--iterations", help="suffix appended to benchmark names")
    p.add_argument("--runs-out", type=Path, default=Path("runs.csv"))
    p.add_argument("--agg-out", type=Path, default=Path("aggregate.csv"))
    p.add_argument("--inspect", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.directory.glob(args.glob))
    paths = [p for p in paths if p.name not in {args.runs_out.name, args.agg_out.name}]
    if not paths:
        print(f"error: no files matching {args.glob} in {args.directory}", file=sys.stderr)
        return 1

    if args.inspect:
        return inspect(paths, args.skip_lines)

    kernel_times = load_kernel_times(args)
    if kernel_times:
        print(f"kernel timings loaded for {len(kernel_times)} run(s)")

    unit_scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(args.time_unit or "")
    runs, skipped = [], []
    reported_columns = False

    for path in paths:
        header, rows, _ = read_csv(path, args.skip_lines)
        if len(rows) < 2:
            skipped.append((path.name, "fewer than 2 rows"))
            continue

        allowed = numeric_columns(header, rows)
        tcol = args.time_col or pick_column(header, TIME_HINTS, allowed)
        pcol = args.power_col or pick_column(header, POWER_HINTS, allowed, POWER_PREFER)
        if not tcol or not pcol:
            skipped.append((path.name, f"could not find time/power columns in {header}"))
            continue

        extra_cols = [c for c in args.extra_cols.split(",") if c and c in header]
        missing = [c for c in args.extra_cols.split(",") if c and c not in header]
        if missing and not reported_columns:
            print(f"note: extra columns not in this file: {', '.join(missing)}")

        pairs, extra_vals = [], {c: [] for c in extra_cols}
        for row in rows:
            try:
                t, p = float(row[tcol]), float(row[pcol])
            except (TypeError, ValueError):
                continue
            pairs.append((t, p))
            for c in extra_cols:
                try:
                    extra_vals[c].append(float(row[c]))
                except (TypeError, ValueError):
                    extra_vals[c].append(float("nan"))
        if len(pairs) < 2:
            skipped.append((path.name, "fewer than 2 parseable samples"))
            continue

        times = [t for t, _ in pairs]
        power = [p for _, p in pairs]
        extra_vals = {c: v for c, v in extra_vals.items()
                      if len(v) == len(times) and not all(math.isnan(x) for x in v)}

        scale, why = (unit_scale, "forced by --time-unit") if unit_scale \
            else guess_time_scale(times, tcol)
        t0 = times[0]
        times = [(t - t0) * scale for t in times]

        if not reported_columns:
            print(f"time column : {tcol}  (scale {scale:g} s/unit, {why})")
            print(f"power column: {pcol}")
            reported_columns = True

        bench, run = parse_name(path)
        kernel_s = lookup_kernel(kernel_times, bench, run)

        clock_win, clock_cov = None, None
        clocks = extra_vals.get(args.clock_col)
        if clocks and args.window_mode in ("clock", "auto"):
            thr = resolve_threshold(args.clock_threshold, clocks)
            found = clock_window(times, clocks, thr, args.clock_min_frac, args.clock_gap)
            if found:
                clock_win, clock_cov = found

        kernel_win = None
        if kernel_s:
            end = times[-1] - args.tail_pad
            kernel_win = (max(end - kernel_s, times[0]), end)

        if args.window_mode == "clock":
            window, source = clock_win, "clock"
        elif args.window_mode == "kernel":
            window, source = kernel_win, "kernel"
        else:  # auto: kernel time is authoritative, clock is the fallback
            window, source = (kernel_win, "kernel") if kernel_win else (clock_win, "clock")
        if window is None:
            source = "none"

        result = integrate(times, power, args.static, window, extra_vals)
        result.update(benchmark=bench, run=run if run is not None else 1,
                      file=path.name, kernel_s=kernel_s or "", window_source=source)
        if clock_win:
            result["clock_window_s"] = clock_win[1] - clock_win[0]
            result["clock_coverage"] = clock_cov
            result["window_frac_of_trace"] = ((clock_win[1] - clock_win[0]) /
                                              (times[-1] - times[0])) if times[-1] > times[0] else 0
            if kernel_s:
                result["clock_vs_kernel"] = (clock_win[1] - clock_win[0]) / kernel_s
        if window and not result["windowed"]:
            skipped.append((path.name, "window held too few samples; used full trace"))
        runs.append(result)

    if not runs:
        print("error: no usable files", file=sys.stderr)
        for name, why in skipped:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1

    fields = ["benchmark", "run", "file", "samples", "duration_s", "kernel_s", "windowed",
              "total_energy_J", "dynamic_energy_J", "mean_power_W",
              "mean_dynamic_power_W", "max_power_W", "min_power_W",
              "window_source", "clock_window_s", "clock_vs_kernel",
              "clock_coverage", "window_frac_of_trace"]
    extra_fields = sorted({k for r in runs for k in r
                           if k.endswith(("_median", "_min", "_max"))
                           and k not in fields and not k.startswith("calculated")})
    fields += extra_fields
    with args.runs_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(runs, key=lambda r: (r["benchmark"], r["run"])):
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    agg = aggregate(runs, args.iterations, extra_fields)
    with args.agg_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agg[0].keys()))
        writer.writeheader()
        writer.writerows(agg)

    print(f"\nwrote {args.runs_out}  ({len(runs)} runs)")
    print(f"wrote {args.agg_out}  ({len(agg)} benchmarks)")
    if skipped:
        print(f"\nskipped {len(skipped)} file(s):", file=sys.stderr)
        for name, why in skipped:
            print(f"  {name}: {why}", file=sys.stderr)

    check(runs, args.static)

    width = max(len(r["benchmark"]) for r in agg)
    has_clock = any("gfx_clock_MHz_median" in r for r in agg)
    header = (f"\n{'benchmark'.ljust(width)}  {'n':>2}  {'dyn_J':>12}  "
              f"{'dur_s':>8}  {'dyn_W':>8}  {'max_W':>7}  win")
    if has_clock:
        header += f"  {'clk_MHz':>16}"
    print(header)
    for row in agg:
        line = (f"{row['benchmark'].ljust(width)}  {row['count_rows']:>2}  "
                f"{float(row['calculated_util_energy_dynamic_median']):>12,.2f}  "
                f"{float(row['execution_time_ms_median']) / 1000:>8.2f}  "
                f"{float(row['mean_dynamic_power_W']):>8.1f}  "
                f"{float(row['max_power_W']):>7.1f}  {row['windowed']:<7}")
        if has_clock and "gfx_clock_MHz_median" in row:
            line += (f"  {row['gfx_clock_MHz_median']:>6} "
                     f"[{row['gfx_clock_MHz_min']}-{row['gfx_clock_MHz_max']}]")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())