#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /home/jvwei/soma-retargeter/.venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/soma-lafan1-warp-cache}"

: "${INPUT_ROOT:=/home/jvwei/datasets/lafan1}"
: "${CSV_ROOT:=/home/jvwei/datasets/lafan1_retargeted/chocolate_csv30}"
: "${NPZ_ROOT:=/home/jvwei/datasets/lafan1_retargeted/chocolate_npz50}"
: "${BATCH_SIZE:=8}"
: "${DEVICE:=cuda:0}"
: "${RESUME:=0}"
: "${SKIP_RETARGET:=0}"

mapfile -t source_files < <(find "$INPUT_ROOT" -maxdepth 1 -type f -name '*.bvh' -print | sort)
if [[ ${#source_files[@]} -eq 0 ]]; then
    echo "No LAFAN1 BVH files found under $INPUT_ROOT" >&2
    exit 1
fi

if [[ "$RESUME" != "1" ]] && {
    { [[ "$SKIP_RETARGET" != "1" ]] && find "$CSV_ROOT" -type f -name '*.csv' -print -quit 2>/dev/null | grep -q .; } ||
    find "$NPZ_ROOT" -type f -name '*.npz' -print -quit 2>/dev/null | grep -q .;
}; then
    echo "Refusing to overwrite existing CSV/NPZ output; set RESUME=1 to validate and continue" >&2
    exit 1
fi

mkdir -p "$CSV_ROOT" "$NPZ_ROOT"
stage_root=$(mktemp -d /tmp/lafan1-export-stage.XXXXXX)
config_path=$(mktemp /tmp/lafan1-export-config.XXXXXX.json)
trap 'rm -rf "$stage_root"; rm -f "$config_path"' EXIT

is_valid_npz() {
    python - "$1" <<'PY'
import sys
import numpy as np

required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w")
try:
    with np.load(sys.argv[1]) as data:
        valid = float(data["fps"]) == 50.0
        valid &= all(name in data and np.isfinite(data[name]).all() for name in required)
        valid &= len(data["joint_pos"]) > 0
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

pending=()
for source_path in "${source_files[@]}"; do
    stem=$(basename "${source_path%.bvh}")
    csv_path="$CSV_ROOT/$stem.csv"
    npz_path="$NPZ_ROOT/$stem.npz"
    if [[ "$RESUME" == "1" && -s "$csv_path" && -f "$npz_path" ]] && is_valid_npz "$npz_path"; then
        echo "Skipping verified output $stem"
        continue
    fi
    if [[ -e "$npz_path" ]] || { [[ "$SKIP_RETARGET" != "1" ]] && [[ -e "$csv_path" ]]; }; then
        echo "Incomplete or invalid existing output for $stem; refusing to overwrite" >&2
        exit 1
    fi
    pending+=("$source_path")
    ln -s "$source_path" "$stage_root/$(basename "$source_path")"
done

if [[ "$SKIP_RETARGET" != "1" && ${#pending[@]} -gt 0 ]]; then
    cat >"$config_path" <<JSON
{
  "import_folder": "$stage_root",
  "export_folder": "$CSV_ROOT",
  "batch_size": $BATCH_SIZE,
  "retargeter": "Newton",
  "retarget_source": "lafan1",
  "retarget_target": "chocolate",
  "retarget_source_facing_direction": "Mujoco"
}
JSON
    cd "$project_root"
    python app/bvh_to_csv_converter.py --config "$config_path" --viewer null --device "$DEVICE"
fi

for source_path in "${source_files[@]}"; do
    stem=$(basename "${source_path%.bvh}")
    csv_path="$CSV_ROOT/$stem.csv"
    npz_path="$NPZ_ROOT/$stem.npz"
    if [[ -f "$npz_path" ]] && is_valid_npz "$npz_path"; then
        continue
    fi
    if [[ ! -s "$csv_path" ]]; then
        echo "Missing retargeted CSV for $stem: $csv_path" >&2
        exit 1
    fi
    frame_time=$(awk '/^[[:space:]]*Frame Time:/ {print $3; exit}' "$source_path")
    input_fps=$(awk -v dt="$frame_time" 'BEGIN {if (dt <= 0) exit 1; printf "%.10g", 1.0/dt}')
    cd "$project_root"
    python app/csv_to_npz.py "$csv_path" "$npz_path" \
        --robot chocolate --input-fps "$input_fps" --output-fps 50
done

echo "Validated ${#source_files[@]} one-to-one LAFAN1 motions; NPZ output is direct source FPS to 50 Hz"
