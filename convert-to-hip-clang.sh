# NOTE: this script will not work as-is
# its a temporary hipify-clang version that only works with shell.nix generating the envvars

# Define the base directories from the accelwattch-ubench collection
BASE_DIRS=("static_power_modeling")

echo "Starting CUDA to HIP conversion and scaling for MI300X..."
for dir in "${BASE_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Warning: Directory '$dir' not found. Skipping..."
        continue
    fi

    find "$dir" -type f -name "*.cu" | while read -r cu_file; do

        if grep -qE '\b(__)?asm(__)?[[:space:]]*\(' "$cu_file"; then
            echo "Skipped: $cu_file (Contains inline assembly)"
            continue
        fi

        dir_name=$(dirname "$cu_file")
        base_name=$(basename "$cu_file" .cu)
        hip_file="$dir_name/$base_name.hip"

        sed -i 's/^#define NUM_OF_BLOCKS.*/#define NUM_OF_BLOCKS (304 * 1024)/g' "$cu_file"

        sed -i 's/(i%32)==0/(i%64)==0/g' "$cu_file"

        if hipify-clang "$cu_file" \
                --clang-resource-directory="$(clang -print-resource-dir)" \
                --cuda-path="$CUDA_PATH" \
                -o "$hip_file" \
                --print-stats \
                -- $CXX_STDLIB_INCLUDES -include ctime -include cstdlib -Wno-register; then
                    # the include args are hacks for now, i think it will generate useless includes
            echo "Converted & Scaled: $cu_file -> $hip_file"
        else
            echo "Failed to convert: $cu_file"
        fi
    done
done
echo "Conversion complete!"
