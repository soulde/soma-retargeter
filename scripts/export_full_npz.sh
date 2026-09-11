#!/usr/bin/env bash
set -euo pipefail

cd /home/jvwei/soma-retargeter
source /home/jvwei/soma-retargeter/.venv/bin/activate

INPUT_ROOT=/home/jvwei/datasets/soma_uniform/filtered/retargeted/full_20260907_1552
OUTPUT_ROOT=/home/jvwei/datasets/soma_uniform/filtered/retargeted/npz_20260907_$(date +%H%M%S)
mkdir -p "$OUTPUT_ROOT"

while IFS= read -r csv; do
    rel=${csv#"$INPUT_ROOT"/}
    robot=${rel%%/*}
    name=${rel##*/}
    stem=${name%.csv}
    mkdir -p "$OUTPUT_ROOT/$robot"
    python app/csv_to_npz.py "$csv" "$OUTPUT_ROOT/$robot/$stem.npz" --robot "$robot" --input-fps 120 --output-fps 120
done < <(find "$INPUT_ROOT" -type f -name '*.csv' | sort)

echo "Exported NPZ files to $OUTPUT_ROOT"
