# OBBSeg_final

OBBSeg_final is an oriented bounding box guided medical image segmentation project based on SAM2. The current training entry uses `torchrun` with DistributedDataParallel, and the evaluation entry reports metrics only. Evaluation visualization and prediction image saving are disabled.

## Directory Layout

```text
OBBSeg_final/
├── dataset/                 # Raw and pre-processed datasets
├── pretrain/                # SAM2 checkpoints, for example sam2_hiera_large.pt
├── sam2/                    # SAM2 implementation
├── sam2_configs/            # SAM2 yaml configs
└── source/
    ├── train.py             # Training entry
    ├── train_torchrun.sh    # Multi-GPU launch script
    ├── test.py              # Evaluation entry
    ├── model.py             # OBBSeg model
    ├── dataset.py           # Dataset loaders
    ├── preprocess.py        # Dataset preprocessing
    └── utils.py             # Losses and OBB utilities
```

## Environment

Use the existing conda environment on the server:

```sh
cd /mnt/d/workstation/ExpHome/OBBMed/OBBSeg_final/source
/root/anaconda3/bin/python --version
```

The launch script uses:

```text
/root/anaconda3/bin/python
/root/anaconda3/bin/torchrun
```

Before training, check GPU visibility:

```sh
nvidia-smi
CUDA_DEVICES=0,1 NPROC_PER_NODE=2 sh train_torchrun.sh --help
```

## Data and Checkpoints

Expected dataset layout before preprocessing:

```text
dataset/<DatasetName>/TrainDataset
dataset/<DatasetName>/TestDataset
```

For `SUN-SEG_Pre`, the code also supports:

```text
TestEasyDataset
TestHardDataset
```

The training script automatically preprocesses the dataset if this directory is missing:

```text
dataset/<DatasetName>-Processed
```

The SAM2 checkpoint path is fixed in the config:

```text
pretrain/sam2_hiera_large.pt
```

## Training

Recommended launch command:

```sh
cd /mnt/d/workstation/ExpHome/OBBMed/OBBSeg_final/source
CUDA_DEVICES=0,1 NPROC_PER_NODE=2 sh train_torchrun.sh
```

Default script parameters:

```text
epoch: 20
lr: 0.001
batchsize: 8
weight_decay: 0.0005
clip: 0.5
num_worker: 4
train_dataset: PraNetDataset
prompt_mode: Box
save_path: SAM2
```

Override parameters by appending normal `train.py` arguments:

```sh
CUDA_DEVICES=0,1 NPROC_PER_NODE=2 sh train_torchrun.sh \
  --epoch 50 \
  --batchsize 8 \
  --train_dataset PraNetDataset \
  --prompt_mode Box \
  --save_path SAM2
```

Supported prompt modes in the current code:

```text
Point
Box
Scribble
Circle
OBB
```

Training outputs:

```text
source/train.log
source/SAM2/<DatasetName>/log/
source/SAM2/<DatasetName>/model-<PromptMode>.pt
```

Only rank 0 writes logs, runs evaluation after each epoch, and saves the best checkpoint.

## Evaluation

Run evaluation after a checkpoint is available:

```sh
cd /mnt/d/workstation/ExpHome/OBBMed/OBBSeg_final/source
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/bin/python test.py \
  --batchsize 64 \
  --num_worker 4 \
  --test_dataset PraNetDataset \
  --prompt_mode Box \
  --load_path SAM2/PraNetDataset
```

Important: `--load_path` should point to the directory containing:

```text
model-<PromptMode>.pt
```

For example, with `--prompt_mode Box`, the evaluation script loads:

```text
SAM2/PraNetDataset/model-Box.pt
```

Evaluation prints:

```text
cnt
mae
dice
iou
hd95
Inference Time
```

The evaluation script no longer saves prediction images or temporary visualization files.

## Common Commands

Use GPU 4 and 5:

```sh
CUDA_DEVICES=4,5 NPROC_PER_NODE=2 sh train_torchrun.sh
```

Use one GPU for quick debugging:

```sh
CUDA_DEVICES=0 NPROC_PER_NODE=1 sh train_torchrun.sh --epoch 1 --batchsize 2 --num_worker 0
```

Use a different distributed port if another job is already using the default:

```sh
MASTER_PORT=29515 CUDA_DEVICES=0,1 NPROC_PER_NODE=2 sh train_torchrun.sh
```

Check active training processes:

```sh
ps -eo pid,ppid,pgid,sid,stat,etime,cmd | grep -E "train_torchrun|torchrun|source/train.py| train.py" | grep -v grep
```

Check recent training log:

```sh
tail -80 train.log
```

## Troubleshooting

If `sh train_torchrun.sh` reports CUDA unavailable, check:

```sh
nvidia-smi
CUDA_DEVICES=0,1 /root/anaconda3/bin/python - <<'PY'
import os
import torch
print(os.environ.get("CUDA_VISIBLE_DEVICES"))
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

If `torchrun` cannot be found, use the launch script instead of running bare `torchrun`; it calls `/root/anaconda3/bin/torchrun` directly.

If another training job is already using port `29500`, set a different `MASTER_PORT`.

## Git Tracking

This directory is an independent git repository. The `.gitignore` is configured to track only:

```text
.py
.sh
.yaml
.gitignore
```

Large data, checkpoints, logs, pycache, and README files are ignored by default.
