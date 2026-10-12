#!/usr/bin/env bash
#
# Profile every ubench with rocprof-compute at the SAME iteration counts used for
# the opcode counts, then write
#   cache_metrics.csv    - every metric from the dumped blocks (for inspection)
#   cache_hit_rates.csv  - benchmark,l1_hit,l2_hit as fractions, one row per
#                          <bench>_<iters>, ready for scale_instruction.ipynb
#
# `profile` re-launches the binary once per counter pass, so its cost is roughly
# (number of passes) x (one full run of the binary). Collecting only the vL1D
# and L2 blocks keeps the number of passes small.
#
# Usage:
#   ./profile_cache.sh                          # all benchmarks
#   ./profile_cache.sh LDGSTG8 LDC16            # named only
#   ITERATIONS="10000" ./profile_cache.sh       # one iteration count

set -uo pipefail

BIN_DIR="${BIN_DIR:-$HOME/accelwattch-ubenches-hip}"
WORK_DIR="${WORK_DIR:-$PWD/cache_workloads}"
ITERATIONS="${ITERATIONS:-1000 10000}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
# CHANGED: 16 = vL1D, 17 = L2/TCC. Both hit rates live in these blocks' own
# speed-of-light subsections (16.1, 17.1), so the 2.1 system summary, which
# pulls counters from nearly every hardware block, is no longer needed.
BLOCKS="${BLOCKS:-16 17}"
# Blocks to COLLECT. Defaults to the blocks you analyze. Some rocprof-compute
# versions want hardware block names here instead, e.g. PROFILE_BLOCKS="TCP TCC"
PROFILE_BLOCKS="${PROFILE_BLOCKS:-$BLOCKS}"

log()  { printf '[%s] %s\n' "$(date +%T)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%T)" "$*" >&2; }

command -v rocprof-compute >/dev/null || { echo "rocprof-compute not in PATH" >&2; exit 1; }
mkdir -p "$WORK_DIR"

block_args=()
if rocprof-compute profile --help 2>&1 | grep -qE -- '(^|[[:space:]])(-b|--block)([[:space:],]|$)'; then
    block_args=(--block $PROFILE_BLOCKS)
    log "profiling only blocks: $PROFILE_BLOCKS"
else
    warn "this rocprof-compute has no profile --block option; collecting everything (slow)"
fi

targets=("$@")
if (( ${#targets[@]} == 0 )); then
    mapfile -t targets < <(find "$BIN_DIR/bin" -maxdepth 1 -type f -executable -printf '%f\n' | sort)
fi
log "${#targets[@]} benchmark(s), ITERATIONS=$ITERATIONS, timeout ${TIMEOUT_SEC}s per profile"

# ------------------------------------------------------------------ profile
for b in "${targets[@]}"; do
    exe="$BIN_DIR/bin/$b"
    [[ -x "$exe" ]] || { warn "skip $b: not executable"; continue; }
    for it in $ITERATIONS; do
        name="${b}_${it}"
        done_flag="$WORK_DIR/${name}.done"
        # A killed or failed run leaves a partial workload directory behind,
        # so skip only runs that finished. Partial ones are redone.
        if [[ -f "$done_flag" ]]; then
            log "$name: already profiled, skipping"
            continue
        fi
        if [[ -d "$WORK_DIR/$name" ]]; then
            warn "$name: removing incomplete workload from an earlier run"
            rm -rf -- "${WORK_DIR:?}/$name"
        fi

        log "profiling $name"
        start=$SECONDS
        timeout -k 30 -s KILL "$TIMEOUT_SEC" \
            # rocprof-compute >= 3.x writes the workload INTO --path (no <name>/<arch>
            # subdirectory), so each run needs its own directory or they overwrite.
            rocprof-compute profile -n "$name" --path "$WORK_DIR/$name" --no-roof "${block_args[@]}" \
                -- "$exe" "$it" \
            > "$WORK_DIR/${name}.profile.log" 2>&1
        rc=$?
        elapsed=$(( SECONDS - start ))
        if (( rc == 0 )) && [[ -f "$WORK_DIR/$name/pmc_perf.csv" ]]; then
            touch "$done_flag"
            log "$name: done in ${elapsed}s"
        elif (( rc == 0 )); then
            warn "$name: profile exited 0 but wrote no $WORK_DIR/$name/pmc_perf.csv"
        elif (( rc == 124 || rc == 137 )); then
            warn "$name: TIMED OUT after ${elapsed}s (raise TIMEOUT_SEC); workload is incomplete"
        else
            warn "$name: profile failed rc=$rc after ${elapsed}s (see $WORK_DIR/${name}.profile.log)"
        fi
    done
done

# ------------------------------------------------------------------ analyze
log "analyzing"
for b in "${targets[@]}"; do
    for it in $ITERATIONS; do
        name="${b}_${it}"
        [[ -f "$WORK_DIR/${name}.done" ]] || continue
        wl="$WORK_DIR/$name"
        [[ -f "$wl/pmc_perf.csv" ]] || { warn "$name: no pmc_perf.csv in $wl, skipping analyze"; continue; }
        rocprof-compute analyze -p "$wl" --block $BLOCKS \
            > "$WORK_DIR/${name}.analyze.txt" 2>&1 \
            || warn "$name: analyze failed"
    done
done

# ------------------------------------------------------------- collect to CSV
log "collecting"
python3 - "$WORK_DIR" << 'PY'
import csv, pathlib, re, sys

work = pathlib.Path(sys.argv[1])
SEP = r"[|\u2502]"
ROW = re.compile(
    rf"^\s*{SEP}\s*([\d.]+)\s*{SEP}\s*([^|\u2502]+?)\s*{SEP}"
    rf"\s*([-\d.eE+]*)\s*{SEP}\s*([^|\u2502]*?)\s*{SEP}"
)

# CHANGED: each hit rate is taken from its block's speed-of-light subsection:
# (metric-ID prefix, metric-name pattern). Edit if your version words or
# numbers them differently -- the script prints what it finds if a pattern misses.
HIT_RATES = {
    "l1_hit": ("16.1.", re.compile(r"hit\s*rate", re.I)),   # vL1D
    "l2_hit": ("17.1.", re.compile(r"hit\s*rate", re.I)),   # L2 / TCC
}

def is_percent(unit):                                        # CHANGED: wider unit check
    u = unit.strip().lower()
    return u.startswith("pct") or u.startswith("percent") or u == "%"

data, names, units = {}, {}, {}
for path in sorted(work.glob("*.analyze.txt")):
    bench = path.name[: -len(".analyze.txt")]          # <bench>_<iters>
    if not (work / f"{bench}.done").exists():
        continue
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
        names[mid], units[mid] = name, unit
    if vals:
        data[bench] = vals

if not data:
    sys.exit("no metrics parsed -- check the .analyze.txt files")

cols = sorted(names, key=lambda s: [int(p) for p in s.split(".")])
with (work / "cache_metrics.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["benchmark"] + [f"{c} {names[c]} ({units[c]})" if units[c] else f"{c} {names[c]}" for c in cols])
    for bench in sorted(data):
        w.writerow([bench] + [data[bench].get(c, "") for c in cols])

chosen = {}
for key, (prefix, pat) in HIT_RATES.items():
    ids = [m for m in names if m.startswith(prefix) and pat.search(names[m])]
    if len(ids) != 1:
        block = prefix.split(".")[0] + "."
        cands = sorted(f"{m} {names[m]} ({units[m]})" for m in names
                       if m.startswith(block) and "hit" in names[m].lower())
        sys.exit(f"{key}: {pat.pattern!r} under {prefix} matched {ids or 'nothing'}.\n"
                 f"hit-rate metrics in block {block[:-1]}: {cands}\n"
                 f"edit HIT_RATES and re-run (analyze is cheap).")
    chosen[key] = ids[0]
    print(f"{key} <- {ids[0]} {names[ids[0]]} ({units[ids[0]]})")

out = work / "cache_hit_rates.csv"
bad = []
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["benchmark"] + list(chosen))
    for bench in sorted(data):
        row = []
        for key, mid in chosen.items():
            v = data[bench].get(mid)
            if v is not None and is_percent(units[mid]):
                v /= 100.0
            if v is not None and not 0.0 <= v <= 1.0:
                bad.append((bench, key, v))
            row.append("" if v is None else f"{v:.6f}")
        w.writerow([bench] + row)
print(f"wrote {out}  ({len(data)} workloads)")
for b, k, v in bad:
    print(f"WARNING: {b} {k} = {v} outside [0,1]", file=sys.stderr)

by_base = {}
for bench in data:
    base, _, it = bench.rpartition("_")
    if it.isdigit():
        by_base.setdefault(base, {})[int(it)] = bench
print("\nhit-rate change from lowest to highest iteration count (|delta| > 0.05 flagged):")
for base in sorted(by_base):
    its = sorted(by_base[base])
    if len(its) < 2:
        continue
    lo, hi = data[by_base[base][its[0]]], data[by_base[base][its[-1]]]
    d = {k: (hi.get(m, 0) - lo.get(m, 0)) / (100.0 if is_percent(units[m]) else 1.0)
         for k, m in chosen.items()}
    flag = "  <--" if any(abs(v) > 0.05 for v in d.values()) else ""
    print(f"  {base:28} " + "  ".join(f"{k} {v:+.3f}" for k, v in d.items()) + flag)
PY
