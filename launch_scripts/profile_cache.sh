#!/usr/bin/env bash
#
# Profile every ubench once with rocprof-compute, then extract cache metrics
# into a single CSV.
#
# `profile` re-runs the kernel once per counter group, so it is expensive and
# is done exactly once per benchmark. `analyze` reads the saved workload and
# costs nothing, so slice the same profile as many ways as you like afterwards.
#
# Usage:
#   ./profile_cache.sh                 # all benchmarks
#   ./profile_cache.sh LDGSTG8 LDC16   # named only
#   ITERATIONS=10 ./profile_cache.sh   # cache rates are ratios; keep this tiny

set -uo pipefail

BIN_DIR="${BIN_DIR:-$HOME/accelwattch-ubenches-hip}"
WORK_DIR="${WORK_DIR:-$PWD/cache_workloads}"
ITERATIONS="${ITERATIONS:-100}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
# Blocks to dump. 2.1 = System Speed-of-Light (has the two headline hit rates),
# 16 = vL1D (all subsections), 17 = L2/TCC (all subsections).
BLOCKS="${BLOCKS:-2.1 16 17}"

log()  { printf '[%s] %s\n' "$(date +%T)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%T)" "$*" >&2; }

command -v rocprof-compute >/dev/null || { echo "rocprof-compute not in PATH" >&2; exit 1; }
mkdir -p "$WORK_DIR"

targets=("$@")
if (( ${#targets[@]} == 0 )); then
    mapfile -t targets < <(find "$BIN_DIR/bin" -maxdepth 1 -type f -executable -printf '%f\n' | sort)
fi
log "${#targets[@]} benchmark(s), ITERATIONS=$ITERATIONS"

# ------------------------------------------------------------------ profile
for b in "${targets[@]}"; do
    exe="$BIN_DIR/bin/$b"
    [[ -x "$exe" ]] || { warn "skip $b: not executable"; continue; }

    if compgen -G "$WORK_DIR/$b/*/" >/dev/null; then
        log "$b: already profiled, skipping"
        continue
    fi

    log "profiling $b"
    timeout -k 30 -s KILL "$TIMEOUT_SEC" \
        rocprof-compute profile -n "$b" --path "$WORK_DIR" --no-roof \
            -- "$exe" "$ITERATIONS" \
        > "$WORK_DIR/${b}.profile.log" 2>&1 \
        || warn "$b: profile failed or timed out (see $WORK_DIR/${b}.profile.log)"
done

# ------------------------------------------------------------------ analyze
log "analyzing"
for b in "${targets[@]}"; do
    # The profile step creates <work>/<name>/<ARCH>/ -- discover the arch dir.
    wl=$(find "$WORK_DIR/$b" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
    [[ -n "$wl" ]] || continue
    rocprof-compute analyze -p "$wl" --block $BLOCKS \
        > "$WORK_DIR/${b}.analyze.txt" 2>&1 \
        || warn "$b: analyze failed"
done

# ------------------------------------------------------------- collect to CSV
log "collecting into cache_metrics.csv"
python3 - "$WORK_DIR" << 'PY'
import csv, pathlib, re, sys

work = pathlib.Path(sys.argv[1])
# rocprof-compute draws its tables with box-drawing characters, not ASCII
# pipes, so accept either:
#   | 2.1.20 | L2 Cache Hit Rate | 35.55 | Pct | 100 | 35.55 |
SEP = r"[|\u2502]"
ROW = re.compile(
    rf"^\s*{SEP}\s*([\d.]+)\s*{SEP}\s*([^|\u2502]+?)\s*{SEP}"
    rf"\s*([-\d.eE+]*)\s*{SEP}\s*([^|\u2502]*?)\s*{SEP}"
)

data, metrics = {}, {}
for path in sorted(work.glob("*.analyze.txt")):
    bench = path.name[: -len(".analyze.txt")]
    vals = {}
    for line in path.read_text(errors="replace").splitlines():
        m = ROW.match(line)
        if not m:
            continue
        mid, name, value, unit = m.groups()
        if not value:
            continue
        try:
            vals[mid] = float(value)
        except ValueError:
            continue
        metrics[mid] = f"{name} ({unit})" if unit else name
    if vals:
        data[bench] = vals

if not data:
    print("no metrics parsed -- check the .analyze.txt files", file=sys.stderr)
    sys.exit(1)

cols = sorted(metrics, key=lambda s: [int(p) for p in s.split(".")])
out = work / "cache_metrics.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["benchmark"] + [f"{c} {metrics[c]}" for c in cols])
    for bench in sorted(data):
        w.writerow([bench] + [data[bench].get(c, "") for c in cols])
print(f"wrote {out}  ({len(data)} benchmarks x {len(cols)} metrics)")

# Console view: just the two headline hit rates.
head = {k: v for k, v in metrics.items() if "Hit Rate" in v}
if head:
    width = max(len(b) for b in data)
    ids = sorted(head, key=lambda s: [int(p) for p in s.split(".")])
    print()
    print("benchmark".ljust(width) + "  " + "  ".join(head[i][:22].rjust(22) for i in ids))
    for bench in sorted(data):
        row = data[bench]
        print(bench.ljust(width) + "  " +
              "  ".join(f"{row[i]:22.2f}" if i in row else " " * 22 for i in ids))
PY