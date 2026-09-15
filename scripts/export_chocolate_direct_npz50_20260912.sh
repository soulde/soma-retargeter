#!/usr/bin/env bash
set -euo pipefail

export SKIP_RETARGET=1
export RESUME=1
export CUDA_VISIBLE_DEVICES=""
export CSV_ROOT=/home/jvwei/datasets/soma_uniform/filtered/retargeted/amp_smooth_20260909/chocolate
export NPZ_ROOT=/home/jvwei/datasets/soma_uniform/filtered/retargeted/amp_smooth_20260909/chocolate_direct_npz50

exec /home/jvwei/soma-retargeter/scripts/regenerate_chocolate_amp_smooth_20260909.sh
