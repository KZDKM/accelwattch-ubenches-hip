# HIP Accelwattch Microbenchmarks

This repository contains [rough] HIP ports of the Accelwattch microbenchmarks
found [here](https://github.com/accel-sim/gpu-app-collection/tree/dev/src/cuda/accelwattch-ubench).

## Directions
To compile these benchmarks, please run `make -f Makefile-hip`. You should
have ROCM installed and point to the correct `hipcc` and/or modify the
Makefile as needed. Note that the `convert-to-hip.sh` script assumes that the
sizing is for the MI300X (where there are 304CUs total, you should adjust the
sizing depending on the GPU you're running your programs on).

If you want to run all of these on the AMD HPCFund (or a similar HPC system),
you should look at `launch_scripts/` for a sbatch submission script.
Note that `run-accelwattch-ubenches` does rely on [rocprofwrap-lt](https://github.com/hal-uw/rocprofwrap) for power measurements.

## TODO
The `.hip` microbenchmarks found in these directories are currently very
roughly ported using `hipify`. To more accurately port the benchmarks over,
the following should be done
* Validate that the current functional/branching benchmarks behave as intended
* Modify the rest of the microbenchmarks to be functionally equivalent to the CUDA ones found in the Accelwattch microbenchmarks linked earlier. To do this, the inline assembly will need to be changed depending on the CDNA ISA used.
