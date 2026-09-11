#!/usr/bin/env bash
set -euo pipefail

cd /home/jvwei/soma-retargeter
source /home/jvwei/soma-retargeter/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CONFIG=/home/jvwei/datasets/soma_uniform/filtered/retargeted/amp_smooth_20260909_chocolate.json
CSV_ROOT=/home/jvwei/datasets/soma_uniform/filtered/retargeted/amp_smooth_20260909/chocolate
PHASE_ROOT=/home/jvwei/datasets/soma_uniform/filtered/retargeted/amp_smooth_20260909/chocolate_phase30
NPZ_ROOT=/home/jvwei/datasets/soma_uniform/filtered/retargeted/amp_smooth_20260909/chocolate_npz50

mkdir -p "$CSV_ROOT" "$PHASE_ROOT" "$NPZ_ROOT"

python app/bvh_to_csv_converter.py \
    --config "$CONFIG" \
    --viewer null \
    --device cuda:0

python app/csv_split_phases.py \
    "$CSV_ROOT" "$PHASE_ROOT" \
    --input-fps 120 \
    --stride 4

mapfile -t csv_files < <(find "$PHASE_ROOT" -type f -name '*.csv' -print | sort)
if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "No phase CSV files were generated" >&2
    exit 1
fi

for csv in "${csv_files[@]}"; do
    relative=${csv#"$PHASE_ROOT"/}
    output="$NPZ_ROOT/${relative%.csv}.npz"
    mkdir -p "$(dirname "$output")"
    python app/csv_to_npz.py "$csv" "$output" \
        --robot chocolate \
        --input-fps 30 \
        --output-fps 50
done

npz_count=$(find "$NPZ_ROOT" -type f -name '*.npz' | wc -l)
echo "Generated ${#csv_files[@]} phase CSV files and $npz_count NPZ files"
