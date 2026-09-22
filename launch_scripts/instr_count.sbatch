#!/bin/bash

#SBATCH -J accelwattch-ubench               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 4                  # Total number of mpi tasks requested
#SBATCH -t 00:30:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi3001x            # Desired partition


# roughly four stages:
#   use rocprofv3 --att to wrap binary to arm thread-trace and generate raw data
#   use trace-decoder to decode the generated .att files into tables
#   disasm the binaries to get the literal opcodes
#   match instruction addr to opcode, then emit the count via the python tooling

# python tooling is mostly done by claude
./instr_count.sh
