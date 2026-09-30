#!/usr/bin/env bash
#
# Trace every ubench with ATT, keep only the stats CSVs, and aggregate them.
#
# The raw .att traces and the ui_output_* viewer directories are deleted after
# each benchmark: only stats*.csv (plus disasm.txt for the address->mnemonic
# join) feeds the aggregator, and the traces are large enough to fill a disk.

# The trace decoder is not installed with ROCm before 7.13; point at the
# prebuilt one.
export ATT_LIBRARY_PATH="${ATT_LIBRARY_PATH:-$HOME/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64}"

set -euo pipefail

BIN_DIR="${BIN_DIR:-/home1/kzdkm/accelwattch-ubenches-hip}"
RESULTS_DIR="${RESULTS_DIR:-/home1/kzdkm/accelwattch_ubench_opcodes_10000_unrolled}"
ITERATIONS="${ITERATIONS:-10000}"
TIMEOUT_SEC="${TIMEOUT_SEC:-6000}"
MCPU="${MCPU:-gfx942}"
GPU_ID="${GPU_ID:-0}"

ATT_TARGET_CU="${ATT_TARGET_CU:-1}"
ATT_SE_MASK="${ATT_SE_MASK:-0x1}"
ATT_SIMD_SELECT="${ATT_SIMD_SELECT:-0xF}"
ATT_BUFFER_SIZE="${ATT_BUFFER_SIZE:-0x6000000}"

# Screen each binary with a tiny run before handing it to the profiler.
PRESCREEN="${PRESCREEN:-0}"
PRESCREEN_ITERS="${PRESCREEN_ITERS:-1}"
PRESCREEN_TIMEOUT="${PRESCREEN_TIMEOUT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGGREGATOR="${SCRIPT_DIR}/aggregate_opcodes.py"

KERNEL_EXCLUDE="${KERNEL_EXCLUDE:-pointers_init|convertFp32ToFp16}"

log()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }
die()  { warn "error: $*"; exit 1; }

check_prereqs() {
    command -v rocprofv3 >/dev/null || die "rocprofv3 not found in PATH"
    command -v llvm-objdump >/dev/null \
        || warn "llvm-objdump not found; opcodes will fall back to raw addresses"
    [[ -d "$BIN_DIR/bin1" ]] || die "no such directory: $BIN_DIR/bin1"
    [[ -f "$AGGREGATOR" ]] || die "aggregator not found: $AGGREGATOR"
    [[ -d "$ATT_LIBRARY_PATH" ]] \
        || warn "ATT_LIBRARY_PATH does not exist: $ATT_LIBRARY_PATH"
}

# Killing the host process does not cancel an in-flight dispatch, so the NEXT
# benchmark blocks on the queue. Wait for the GPU to drain before continuing.
wait_gpu_free() {
    local deadline=$(( SECONDS + ${1:-120} ))
    while (( SECONDS < deadline )); do
        rocm-smi --showpids 2>/dev/null | grep -qE '^[0-9]+[[:space:]]' || return 0
        sleep 2
    done
    warn "  GPU still busy after ${1:-120}s"
    return 1
}

recover_gpu() {
    pkill -KILL -f rocprofv3 2>/dev/null || true
    wait_gpu_free 120 || {
        warn "  attempting GPU reset (likely needs privileges)"
        rocm-smi --gpureset -d "$GPU_ID" >/dev/null 2>&1 || true
        sleep 10
    }
}

disassemble() {
    local exe="$1" outdir="$2"
    command -v llvm-objdump >/dev/null || return 0
    llvm-objdump -d --arch-name=amdgcn --mcpu="$MCPU" "$exe" \
        > "${outdir}/disasm.txt" 2>"${outdir}/disasm.err" || {
        warn "  disassembly failed; see ${outdir}/disasm.err"
        rm -f "${outdir}/disasm.txt"
    }
}

# Delete the raw trace and the viewer JSON. Refuses to delete anything if the
# run produced nothing usable, so a failure stays inspectable.
cleanup_trace() {
    local outdir="$1" before after stats att_bytes cap

    stats=$(find "$outdir" -name 'stats*.csv' -size +0c 2>/dev/null | wc -l)
    if (( stats == 0 )) && [[ ! -s "$outdir/disasm.txt" ]]; then
        warn "  nothing usable produced -- keeping raw output for inspection"
        return
    fi

    # A trace pinned at the buffer size was truncated: the opcode mix is then
    # biased toward whatever executed first.
    cap=$(( ATT_BUFFER_SIZE ))
    while read -r att_bytes; do
        (( att_bytes >= cap )) && warn "  WARNING: .att hit the ${ATT_BUFFER_SIZE} buffer cap -- trace truncated, lower ITERATIONS"
    done < <(find "$outdir" -name '*.att' -printf '%s\n' 2>/dev/null)

    before=$(du -sm "$outdir" 2>/dev/null | cut -f1)
    find "$outdir" -type f -name '*.att' -delete 2>/dev/null || true
    find "$outdir" -type d -name 'ui_output_*' -prune -exec rm -rf {} + 2>/dev/null || true
    after=$(du -sm "$outdir" 2>/dev/null | cut -f1)

    log "  freed $(( ${before:-0} - ${after:-0} )) MB; kept ${stats} stats file(s)"
}

# A broken binary should not cost a multi-pass profiling run.
prescreen() {
    local exe="$1"
    (( PRESCREEN )) || return 0
    timeout -k 10 -s KILL "$PRESCREEN_TIMEOUT" "$exe" "$PRESCREEN_ITERS" \
        >/dev/null 2>&1 && return 0
    local rc=$?
    warn "  fails standalone at ${PRESCREEN_ITERS} iterations (exit ${rc}) -- skipping"
    recover_gpu
    return 1
}

profile_one() {
    local exe="$1"
    local ubench outdir rc
    ubench="$(basename "$exe")"
    outdir="${RESULTS_DIR}/${ubench}"

    log "tracing ${ubench}"
    prescreen "$exe" || return 0
    mkdir -p "$outdir"

    local -a att_args=(
        --att
        --att-target-cu "$ATT_TARGET_CU"
        --att-shader-engine-mask "$ATT_SE_MASK"
        --att-simd-select "$ATT_SIMD_SELECT"
        --att-buffer-size "$ATT_BUFFER_SIZE"
    )
    [[ -n "${ATT_LIBRARY_PATH:-}" ]] && att_args+=(--att-library-path "$ATT_LIBRARY_PATH")
    [[ -n "${KERNEL_EXCLUDE:-}" ]]   && att_args+=(--kernel-exclude-regex "$KERNEL_EXCLUDE")

    rc=0
    # setsid gives it its own process group; -k 30 -s KILL escalates past a
    # SIGTERM that rocprofv3 may ignore while waiting on the GPU.
    setsid timeout -k 30 -s KILL "$TIMEOUT_SEC" rocprofv3 \
        "${att_args[@]}" \
        --output-format csv \
        -d "$outdir" \
        -o "$ubench" \
        -- "$exe" "$ITERATIONS" \
        > "${outdir}/stdout.log" 2>&1 || rc=$?

    case $rc in
        0)       disassemble "$exe" "$outdir" ;;
        124|137) warn "  timeout after ${TIMEOUT_SEC}s"; recover_gpu ;;
        *)       warn "  failed (exit ${rc}); see ${outdir}/stdout.log"; recover_gpu ;;
    esac

    cleanup_trace "$outdir"
    (( rc == 0 )) && log "  ok"
    return 0
}

collect_targets() {
    if (( $# > 0 )); then
        local name
        for name in "$@"; do
            if [[ -x "$BIN_DIR/bin1/$name" ]]; then
                printf '%s\n' "$BIN_DIR/bin1/$name"
            else
                warn "skipping ${name}: not an executable in $BIN_DIR/bin"
            fi
        done
    else
        local exe
        for exe in "$BIN_DIR"/bin1/*; do
            [[ -f "$exe" && -x "$exe" ]] && printf '%s\n' "$exe"
        done
    fi
}

main() {
    check_prereqs
    mkdir -p "$RESULTS_DIR"

    local -a targets=()
    mapfile -t targets < <(collect_targets "$@")
    (( ${#targets[@]} > 0 )) || die "no executables to profile"
    log "${#targets[@]} benchmark(s), ITERATIONS=$ITERATIONS"

    local exe
    for exe in "${targets[@]}"; do
        profile_one "$exe"
    done

    log ""
    log "results dir is now $(du -sh "$RESULTS_DIR" 2>/dev/null | cut -f1)"
    python3 "$AGGREGATOR" "$RESULTS_DIR" \
            --output "${RESULTS_DIR}/opcode_counts.csv"
}

main "$@"