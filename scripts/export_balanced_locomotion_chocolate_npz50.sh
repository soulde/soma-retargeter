#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /home/jvwei/soma-retargeter/.venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/soma-balanced-locomotion-warp-cache}"

: "${INPUT_ROOT:=/home/jvwei/datasets/soma_uniform/selected/locomotion_balanced_20260916}"
: "${OUTPUT_ROOT:=/home/jvwei/datasets/soma_uniform/filtered/retargeted/locomotion_balanced_20260916}"
: "${CSV_ROOT:=$OUTPUT_ROOT/chocolate_csv}"
: "${NPZ_ROOT:=$OUTPUT_ROOT/chocolate_npz50}"
: "${BATCH_SIZE:=4}"
: "${DEVICE:=cuda:0}"
: "${RESUME:=0}"

mapfile -t source_files < <(find "$INPUT_ROOT" -maxdepth 1 -type f -name '*.bvh' -print | sort)
if [[ ${#source_files[@]} -eq 0 ]]; then
    echo "No selected SOMA BVH files found under $INPUT_ROOT" >&2
    exit 1
fi

if [[ "$RESUME" != "1" ]] && {
    find "$CSV_ROOT" -type f -name '*.csv' -print -quit 2>/dev/null | grep -q . ||
    find "$NPZ_ROOT" -type f -name '*.npz' -print -quit 2>/dev/null | grep -q .;
}; then
    echo "Refusing to overwrite existing output; set RESUME=1 to continue safely" >&2
    exit 1
fi

mkdir -p "$CSV_ROOT" "$NPZ_ROOT"
config_path=$(mktemp /tmp/soma-balanced-locomotion.XXXXXX.json)
trap 'rm -f "$config_path"' EXIT
python - "$INPUT_ROOT" "$CSV_ROOT" "$BATCH_SIZE" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

input_root, csv_root, batch_size, config_path = sys.argv[1:]
config = {
    "import_folder": input_root,
    "export_folder": csv_root,
    "batch_size": int(batch_size),
    "retargeter": "Newton",
    "retarget_source": "soma",
    "retarget_target": "chocolate",
    "retarget_source_facing_direction": "Mujoco",
}
Path(config_path).write_text(json.dumps(config, indent=2) + "\n")
PY

pending=0
for source_path in "${source_files[@]}"; do
    stem=$(basename "${source_path%.bvh}")
    if [[ ! -s "$CSV_ROOT/$stem.csv" ]]; then
        pending=$((pending + 1))
    fi
done
if (( pending > 0 )); then
    cd "$project_root"
    python app/bvh_to_csv_converter.py \
        --config "$config_path" --viewer null --device "$DEVICE"
fi

for source_path in "${source_files[@]}"; do
    stem=$(basename "${source_path%.bvh}")
    csv_path="$CSV_ROOT/$stem.csv"
    npz_path="$NPZ_ROOT/$stem.npz"
    if [[ -s "$npz_path" && "$RESUME" == "1" ]]; then
        echo "Keeping existing NPZ $stem"
        continue
    fi
    if [[ -e "$npz_path" ]]; then
        echo "Refusing to overwrite $npz_path" >&2
        exit 1
    fi
    if [[ ! -s "$csv_path" ]]; then
        echo "Missing retargeted CSV for $stem" >&2
        exit 1
    fi
    frame_time=$(awk '/^[[:space:]]*Frame Time:/ {print $3; exit}' "$source_path")
    input_fps=$(awk -v dt="$frame_time" 'BEGIN {if (dt <= 0) exit 1; printf "%.10g", 1.0/dt}')
    cd "$project_root"
    python app/csv_to_npz.py "$csv_path" "$npz_path" \
        --robot chocolate --input-fps "$input_fps" --output-fps 50
done

python "$project_root/scripts/report_locomotion_npz_quality.py" \
    "$NPZ_ROOT" "$OUTPUT_ROOT/quality_report.csv" \
    --summary "$OUTPUT_ROOT/quality_summary.json"

echo "Converted ${#source_files[@]} selected SOMA motions; all NPZ files retained"
