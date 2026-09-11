#!/usr/bin/env bash
set -euo pipefail
cd /home/jvwei/soma-retargeter
source /home/jvwei/soma-retargeter/.venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python app/bvh_to_csv_converter.py --config /home/jvwei/datasets/soma_uniform/filtered/retargeted/full_20260907_1552_chocolate.json --viewer null --device cuda:0
python app/bvh_to_csv_converter.py --config /home/jvwei/datasets/soma_uniform/filtered/retargeted/full_20260907_1552_dr02.json --viewer null --device cuda:0
