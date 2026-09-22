#!/bin/bash

# Change below to be where you want results kept
RESULTS_DIR="/home1/kzdkm/accelwattch_ubench_results"
# Change below to be where the parent directory of your accelwattch ubench
# repo (i.e., functional_benchmarks/ and such should be under here)
BIN_DIR="/home1/kzdkm/accelwattch-ubenches-hip"
ITERATIONS=1000000
CORES=608

# Location of rocprofwrap-lt tool.
ROCPROFWRAP_DIR=/home1/kzdkm/rocprofwrap

if [ $# -lt 1 ]; then
    echo "Usage: $0 <ubench-name>"
    exit 1
fi

ubench="$1"
exe="${BIN_DIR}/bin/${ubench}"

make -C ${ROCPROFWRAP_DIR}/rocprofwrap_lt
#make -C ${BIN_DIR} -f Makefile-hip

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Creating results dir! $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
fi

if [ ! -f "$exe" ] || [ ! -x "$exe" ]; then
    echo "Error: $exe does not exist or is not executable"
    exit 1
fi

for run in {1..5}; do
    WRAPPER_CMD=(
        python3 "${ROCPROFWRAP_DIR}/rocprofwrap_lt/wrapper.py"
        --devices 0
        --prefix="${ubench}_profiled_run${run}.csv"
        --
        "$exe" "$ITERATIONS"
    )
    echo "Running ${ubench} (run ${run}/5)"
    # measure runtime of bench
    start=$(date +%s)
    timeout 300 "${WRAPPER_CMD[@]}"
    ret=$?
    end=$(date +%s)
    duration=$((end - start))
    if [ $ret -eq 124 ]; then
        echo "Run $run timed out after 5 minutes, stopping further runs"
        break
    elif [ $ret -ne 0 ]; then
        echo "Run $run failed with exit code $ret, stopping further runs"
        break
    fi
    echo "Run $run succeeded, duration ${duration}s"
    echo "Output to ${RESULTS_DIR}/${ubench}_profiled_run${run}.csv"
    if [ $run -lt 5 ]; then
        if [ $duration -lt 5 ]; then
            echo "Duration <5s, waiting 10 secs before next run..."
            sleep 10
        else
            echo "Waiting 30 seconds before next run..."
            sleep 30
        fi
    fi
done