# Change below to be where you want results kept
RESULTS_DIR="/home1/wmadsmith/accelwattch_ubench_results"
# Change below to be where the parent directory of your accelwattch ubench
# repo (i.e., functional_benchmarks/ and such should be under here)
BIN_DIR="../"
ITERATIONS=1000000

# Location of rocprofwrap-lt tool.
ROCPROFWRAP_DIR=/home1/wmadsmith/rocprofwrap
make -C ${ROCPROFWRAP_DIR}/rocprofwrap_lt
make -C ${BIN_DIR} -f Makefile-hip
if [[ -d "$RESULTS_DIR" ]]; then
        echo "Creating results dir! $RESULTS_DIR"
        mkdir -p $RESULTS_DIR
fi

# Iterate through every file in the directory
for exe in "$BIN_DIR"/bin/*; do
    ubench=$(basename "$exe")
    WRAPPER_CMD=(
     python3 "${ROCPROFWRAP_DIR}/rocprofwrap_lt/wrapper.py" 
     --devices 0 
     --prefix="${ubench}_profiled.csv" 
     --
     "$exe" "$ITERATIONS"
    )
    # Verify the file exists and has executable permissions
    if [ -f "$exe" ] && [ -x "$exe" ]; then
	    if "${WRAPPER_CMD[@]}"; then
		    echo "Executing ${WRAPPER_CMD[*]}"
	    else
		    echo "--- Profiling failed with exit code $? ---"
	    fi
    fi
    echo "Output to ${RESULTS_DIR}/${ubench}_profiled.csv"
done

