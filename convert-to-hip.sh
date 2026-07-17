#!/bin/bash

# Define the base directories from the accelwattch-ubench collection
BASE_DIRS=("cache_and_data_width_mem_benchmakrs")

echo "Starting CUDA to HIP conversion and scaling for MI300X..."

for dir in "${BASE_DIRS[@]}"; do
    # Check if the directory exists before attempting to process
    if [ ! -d "$dir" ]; then
        echo "Warning: Directory '$dir' not found. Skipping..."
        continue
    fi
    
    # Find all .cu files and process them
    find "$dir" -type f -name "*.cu" | while read -r cu_file; do
        
        # --- NEW CHECK: Skip files containing inline assembly ---
        # Matches 'asm (' or '__asm__ (' with or without spaces
        if grep -qE '\b(__)?asm(__)?[[:space:]]*\(' "$cu_file"; then
            echo "Skipped: $cu_file (Contains inline assembly)"
            continue
        fi
        # --------------------------------------------------------

        dir_name=$(dirname "$cu_file")
        base_name=$(basename "$cu_file" .cu)
        hip_file="$dir_name/$base_name.hip_p"
        
        # 1. Update the block sizing to saturate the MI300X (304 CUs)
        # This replaces any line starting with #define NUM_OF_BLOCKS
        sed -i 's/^#define NUM_OF_BLOCKS.*/#define NUM_OF_BLOCKS (304 * 1024)/g' "$cu_file"
        
        # 2. Update warp divergence logic from NVIDIA's 32 to AMD's 64
        sed -i 's/(i%32)==0/(i%64)==0/g' "$cu_file"
        
        # 3. Run hipify-perl and output to the new .hip file
        if hipify-perl "$cu_file" --print-stat > "$hip_file"; then
            echo "Converted & Scaled: $cu_file -> $hip_file"
        else
            echo "Failed to convert: $cu_file"
        fi
    done
done

echo "Conversion complete!"
