#!/usr/bin/env bash
set -euo pipefail

cd /home/jvwei/soma-retargeter
source /home/jvwei/soma-retargeter/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
: "${WARP_CACHE_PATH:=/tmp/soma-retargeter-warp-cache}"
export WARP_CACHE_PATH

: "${CONFIG:=/home/jvwei/datasets/soma_uniform/filtered/retargeted/amp_smooth_20260909_chocolate.json}"
: "${CSV_ROOT:=/home/jvwei/datasets/soma_uniform/filtered/retargeted/amp_smooth_20260909/chocolate}"
: "${NPZ_ROOT:=/home/jvwei/datasets/soma_uniform/filtered/retargeted/amp_smooth_20260909/chocolate_direct_npz50}"
: "${SKIP_RETARGET:=0}"
: "${RESUME:=0}"

mkdir -p "$CSV_ROOT" "$NPZ_ROOT"

if [[ "$SKIP_RETARGET" != "1" ]]; then
    python app/bvh_to_csv_converter.py \
        --config "$CONFIG" \
        --viewer null \
        --device cuda:0
fi

mapfile -t csv_files < <(find "$CSV_ROOT" -type f -name '*.csv' -print | sort)
if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "No CSV files were found under $CSV_ROOT" >&2
    exit 1
fi

for csv in "${csv_files[@]}"; do
    relative=${csv#"$CSV_ROOT"/}
    output="$NPZ_ROOT/${relative%.csv}.npz"
    if [[ -e "$output" ]]; then
        if [[ "$RESUME" == "1" ]]; then
            echo "Skipping existing $output"
            continue
        fi
        echo "Refusing to overwrite $output" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$output")"
    python app/csv_to_npz.py "$csv" "$output" \
        --robot chocolate \
        --input-fps 120 \
        --output-fps 50
done

npz_count=$(find "$NPZ_ROOT" -type f -name '*.npz' | wc -l)
echo "Generated $npz_count direct 120 Hz to 50 Hz NPZ files from ${#csv_files[@]} CSV files"
