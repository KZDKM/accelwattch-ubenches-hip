#!/usr/bin/env bash
#
# Profile every ubench with rocprof-compute at the SAME iteration counts used for
# the opcode counts, then write
#   cache_metrics.csv    - every metric from the dumped blocks (for inspection)
#   cache_hit_rates.csv  - benchmark,l1_hit,l2_hit as fractions, one row per
#                          <bench>_<iters>, ready for scale_instruction.ipynb
#
# `profile` re-launches the binary once per counter pass, so its cost is roughly
# (number of passes) x (one full run of the binary, host setup included).
# Restricting the profiled blocks cuts the number of passes.
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
# 2.1 = System Speed-of-Light (headline hit rates), 16 = vL1D, 17 = L2/TCC.
BLOCKS="${BLOCKS:-2.1 16 17}"
# CHANGED: blocks to COLLECT. Defaults to the blocks you analyze. Some
# rocprof-compute versions want hardware block names here instead, e.g.
#   PROFILE_BLOCKS="SQ TCP TCC"
PROFILE_BLOCKS="${PROFILE_BLOCKS:-$BLOCKS}"

log()  { printf '[%s] %s\n' "$(date +%T)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%T)" "$*" >&2; }

command -v rocprof-compute >/dev/null || { echo "rocprof-compute not in PATH" >&2; exit 1; }
mkdir -p "$WORK_DIR"

# CHANGED: only filter at profile time if this version supports it.
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
        # CHANGED: a killed or failed run leaves a partial workload directory
        # behind, so skip only runs that finished. Partial ones are redone.
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
            rocprof-compute profile -n "$name" --path "$WORK_DIR" --no-roof "${block_args[@]}" \
                -- "$exe" "$it" \
            > "$WORK_DIR/${name}.profile.log" 2>&1
        rc=$?
        elapsed=$(( SECONDS - start ))
        if (( rc == 0 )); then
            touch "$done_flag"
            log "$name: done in ${elapsed}s"
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
        [[ -f "$WORK_DIR/${name}.done" ]] || continue      # CHANGED: complete workloads only
        wl=$(find "$WORK_DIR/$name" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
        [[ -n "$wl" ]] || continue
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

# Pick the two hit rates by NAME inside block 2.1, not by metric ID.
# If rocprof-compute words them differently in your version, edit these.
HIT_RATES = {
    "l1_hit": re.compile(r"^vL1D Cache Hit Rate$", re.I),
    "l2_hit": re.compile(r"^L2 Cache Hit Rate$",   re.I),
}

data, names, units = {}, {}, {}
for path in sorted(work.glob("*.analyze.txt")):
    bench = path.name[: -len(".analyze.txt")]          # <bench>_<iters>
    if not (work / f"{bench}.done").exists():          # CHANGED: stale/partial runs
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
for key, pat in HIT_RATES.items():
    ids = [m for m in names if m.startswith("2.1.") and pat.search(names[m])]
    if len(ids) != 1:
        cands = sorted(f"{m} {names[m]}" for m in names if m.startswith("2.1.") and "hit" in names[m].lower())
        sys.exit(f"{key}: pattern {pat.pattern!r} matched {ids or 'nothing'} in block 2.1.\n"
                 f"hit-rate metrics present: {cands}\nedit HIT_RATES and re-run (analyze is cheap).")
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
            if v is not None and units[mid].lower().startswith("pct"):
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
    d = {k: (hi.get(m, 0) - lo.get(m, 0)) / (100.0 if units[m].lower().startswith("pct") else 1.0)
         for k, m in chosen.items()}
    flag = "  <--" if any(abs(v) > 0.05 for v in d.values()) else ""
    print(f"  {base:28} " + "  ".join(f"{k} {v:+.3f}" for k, v in d.items()) + flag)
PY
