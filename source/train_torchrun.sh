#!/bin/sh
set -eu

cd "$(dirname "$0")"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/anaconda3/bin/torchrun}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

"${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
count = torch.cuda.device_count()
if count == 0:
    print(f"CUDA is not available. CUDA_VISIBLE_DEVICES={devices}", file=sys.stderr)
    print("Check nvidia-smi first; the driver/NVML may need a reset.", file=sys.stderr)
    sys.exit(1)
print(f"CUDA_VISIBLE_DEVICES={devices}; torch sees {count} device(s).")
PY

"${TORCHRUN_BIN}" \
  --master_port "${MASTER_PORT}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  train.py \
  --epoch 50 \
  --lr 0.001 \
  --batchsize 10 \
  --weight_decay 0.0005 \
  --clip 0.5 \
  --num_worker 4 \
  --train_dataset FSPD-Dataset \
  --prompt_mode Point \
  --save_path SAM2 \
  "$@"
