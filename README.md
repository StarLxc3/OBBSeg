# OBBSeg

OBBSeg is a medical image segmentation project using prompt masks such as Box, Point, Scribble, and Circle. The training entry uses `torchrun` with DistributedDataParallel. The evaluation entry reports metrics only and does not save prediction visualizations.

All paths below are relative to a cloned project directory:

```sh
cd OBBSeg
```

## Project Layout

```text
OBBSeg/
├── dataset/
├── pretrain/
├── sam2/
├── sam2_configs/
└── source/
    ├── train.py
    ├── train_torchrun.sh
    ├── test.py
    ├── model.py
    ├── dataset.py
    ├── preprocess.py
    └── utils.py
```

## Environment Setup

Create a new conda environment and install the dependencies listed in `source/requirements.txt`:

```sh
conda create -n obbseg python=3.11 -y
conda activate obbseg
pip install -r source/requirements.txt
```

## Pretrained Checkpoint and SAM2 Config

Place the SAM2 checkpoint under:

```text
pretrain/
```

Different SAM2 variants can be used, such as tiny, small, base-plus, or large. Select the matching yaml file from:

```text
sam2_configs/
├── sam2_hiera_t.yaml
├── sam2_hiera_s.yaml
├── sam2_hiera_b+.yaml
└── sam2_hiera_l.yaml
```

The checkpoint path and corresponding SAM2 config should match the selected backbone variant. They are configured in `source/train.py`, `source/test.py`, and the SAM2 build/config code.

## Dataset Format and Preprocessing

The training code expects preprocessed datasets in this target structure:

```text
dataset/<DatasetName>-Processed/
├── TrainDataset/
│   ├── Frame/
│   ├── GT/
│   ├── OBB/
│   ├── Box/
│   ├── Coord/
│   ├── Scribble/
│   ├── Circle/
│   └── Point/
└── TestDataset/
    ├── Frame/
    ├── GT/
    ├── OBB/
    ├── Box/
    ├── Coord/
    ├── Scribble/
    ├── Circle/
    └── Point/
```

`Coord/` contains OBB coordinate text files. Each line stores one OBB as four points:

```text
x1,y1;x2,y2;x3,y3;x4,y4
```

The current project supports FSPD (`PraNetDataset`) and SUN-SEG. For other datasets, use `source/preprocess.py` to convert your raw dataset into the target `*-Processed` structure above before training or testing.

For SUN-SEG, the code also supports these processed test splits:

```text
TestEasyDataset/
TestHardDataset/
TestDataset/
```

If `dataset/<DatasetName>-Processed` does not exist, `source/train.py` will try to run preprocessing automatically from the corresponding raw dataset directory.

## Training

Enter the source directory:

```sh
cd source
```

Launch multi-GPU training:

```sh
CUDA_DEVICES=0,1 NPROC_PER_NODE=2 sh train_torchrun.sh
```

Default arguments in `source/train_torchrun.sh`:

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

Override training arguments by appending normal `train.py` arguments:

```sh
CUDA_DEVICES=0,1 NPROC_PER_NODE=2 sh train_torchrun.sh \
  --epoch 50 \
  --batchsize 8 \
  --train_dataset PraNetDataset \
  --prompt_mode Box \
  --save_path SAM2
```

If you use a newly created conda environment, point the launch script to that environment:

```sh
PYTHON_BIN="$(which python)" \
TORCHRUN_BIN="$(which torchrun)" \
CUDA_DEVICES=0,1 \
NPROC_PER_NODE=2 \
sh train_torchrun.sh
```

Supported prompt modes:

```text
Point
Box
Scribble
Circle
```

Training outputs:

```text
source/train.log
source/SAM2/<DatasetName>/log/
source/SAM2/<DatasetName>/model-<PromptMode>.pt
```

Only rank 0 writes logs, runs epoch-end evaluation, and saves the best checkpoint.

## Evaluation

Enter the source directory:

```sh
cd source
```

Run evaluation:

```sh
CUDA_VISIBLE_DEVICES=0 python test.py \
  --batchsize 64 \
  --num_worker 4 \
  --test_dataset PraNetDataset \
  --prompt_mode Box \
  --load_path SAM2/PraNetDataset
```

`--load_path` should point to the directory containing:

```text
model-<PromptMode>.pt
```

For example, with `--prompt_mode Box`, the script loads:

```text
SAM2/PraNetDataset/model-Box.pt
```

Eval metrics:

```text
mae
dice
iou
hd95
Inference Time
```

The evaluation script does not save prediction images or visualization files.
