#!/usr/bin/env bash
set -euo pipefail

: "${POLL_INTERVAL:=300}"
export RESUME="${RESUME:-1}"

while true; do
    compute_processes=$(nvidia-smi \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null || true)
    if [[ -z "$compute_processes" ]]; then
        echo "$(date --iso-8601=seconds) GPU has no compute processes; starting LAFAN1 export"
        break
    fi
    echo "$(date --iso-8601=seconds) waiting for GPU compute processes:"
    printf '%s\n' "$compute_processes"
    sleep "$POLL_INTERVAL"
done

exec /home/jvwei/soma-retargeter/scripts/export_lafan1_chocolate_npz50.sh
