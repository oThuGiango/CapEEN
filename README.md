# Image Captioning with Early Exit (CapEEN)

This README matches the current code in `imgcap.py` and the dependency file
`requirements.txt`.

## One-command setup (Jetson Orin / JetPack 6.1)

Run:

```bash
chmod +x setup_jetson.sh
./setup_jetson.sh
```

The script installs:
- system dependencies (`default-jre`, build tools, etc.)
- `cuSPARSELt` runtime needed by Jetson PyTorch wheel
- Python virtual environment (`venv`)
- all Python packages from `requirements.txt`

Then run:

```bash
source venv/bin/activate
python imgcap.py
```

## Why not only requirements.txt?

`requirements.txt` only installs Python packages via pip. Some required pieces
for Jetson (like CUDA runtime libraries such as `libcusparseLt.so.0`) are
system-level libraries, so they must be installed by apt/script, not pip.

## Overview

The script trains an image-captioning pipeline with two phases:

| Phase | What runs | Output |
|-------|-----------|--------|
| Phase 1 | Fine-tune `Swin-Base + GPT-2` end-to-end | Best baseline checkpoint |
| Phase 2 | Freeze the best baseline and train 12 intermediate exit heads | Best exit-head weights |

Architecture:

```text
image -> Swin-Base encoder -> GPT-2 decoder with cross-attention
                                  |
                                  +-> exit head at decoder layer 0
                                  +-> exit head at decoder layer 1
                                  ...
                                  +-> exit head at decoder layer 11
```

During early-exit inference, the decoder generates token by token. For each
token, layers are checked sequentially from 0 to 11. After each decoder layer,
the corresponding exit head predicts the next token. If aggregated confidence
reaches `EXIT_THRESHOLD`, that token exits immediately at the current layer
(so that token does not continue to deeper layers). For the next token,
checking starts again from layer 0 and may run up to layer 11 if needed. If no
exit head reaches the threshold, the script falls back to the baseline decoder
final LM head.

Knowledge distillation and cache are separate:

- Knowledge distillation happens during Phase 2 training. Exit heads learn from
  ground-truth tokens and the teacher model final logits.
- Early-exit inference currently does not use KV cache. Each token recomputes
  the current generated prefix so every decoder layer sees a consistent
  sequence even when the previous token exited early.

## Current Code Configuration

Edit these values near the top of `imgcap.py` before running:

```python
DATASET_ROOT = "dataset"
BASELINE_CKPT_BASE = "./image-captioning"
EXIT_CKPT_DIR_BASE = "./checkpoint/intermediate_head_weights"
DEV_MODE = True
```

Mode-specific paths are derived automatically:

```text
DEV_MODE=True:
  baseline checkpoint -> ./image-captioning-dev
  exit heads          -> ./checkpoint/intermediate_head_weights_dev
  results             -> results/dev/<timestamp>/

DEV_MODE=False:
  baseline checkpoint -> ./image-captioning-full
  exit heads          -> ./checkpoint/intermediate_head_weights_full
  results             -> results/full/<timestamp>/
```

## Dataset Layout

The current code uses **COCO train2014 images** for training and
**COCO val2017 images** for validation/testing.
Annotations are expected in a top-level `annotations/` folder, not inside
`dataset/`.

Expected folder layout:

```text
.
├── annotations/
│   ├── captions_train2014.json
│   └── captions_val2017.json
└── dataset/
    ├── train2014/
    │   └── COCO_train2014_*.jpg
    └── val2017/
        └── COCO_val2017_*.jpg
```

Relevant code:

```python
ANNOTATIONS_DIR = "annotations"
TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "train2014")
VAL_IMAGE_DIR = os.path.join(DATASET_ROOT, "val2017")

train_ann_file = os.path.join(ANNOTATIONS_DIR, "captions_train2014.json")
val_ann_file = os.path.join(ANNOTATIONS_DIR, "captions_val2017.json")
```

Split behavior:

| Split | Source | Approx images | Note |
|-------|--------|---------------|------|
| Train | `dataset/train2014` | ~83k | Uses 3 captions/image, every `(image, caption)` pair is one training sample |
| Val | `dataset/val2017` | ~4.5k | Uses 3 captions/image from 90% of shuffled val2017 image ids |
| Test | `dataset/val2017` | ~0.5k | Keeps all available captions, usually 5 references/image |

`DEV_MODE=True` truncates this to:

```text
200 train samples / 50 val samples / 30 test images
```

## Download Dataset

```bash
mkdir -p dataset annotations

# COCO train2014 annotations: captions_train2014.json
aria2c -x 8 -s 8 -c -d . -o annotations_trainval2014.zip \
  http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip -q annotations_trainval2014.zip annotations/captions_train2014.json -d .
rm -f annotations_trainval2014.zip

# COCO 2017 annotations: captions_val2017.json
aria2c -x 8 -s 8 -c -d . -o annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip annotations/captions_val2017.json -d .
rm -f annotations_trainval2017.zip

# COCO train2014 images
aria2c -x 16 -s 16 -k 1M -c -d dataset -o train2014.zip \
  http://images.cocodataset.org/zips/train2014.zip
unzip -q dataset/train2014.zip -d dataset
rm -f dataset/train2014.zip

# COCO 2017 val images
aria2c -x 16 -s 16 -k 1M -c -d dataset -o val2017.zip \
  http://images.cocodataset.org/zips/val2017.zip
unzip -q dataset/val2017.zip -d dataset
rm -f dataset/val2017.zip
```

## Manual install (alternative)

```bash
pip install -r requirements.txt
sudo apt-get install -y default-jre
```

Notes:

- `pycocoevalcap` is already in `requirements.txt`; do not install it a second time unless the first install fails.
- METEOR requires Java, hence `default-jre`.

## Run

```bash
python imgcap.py
```

Recommended workflow:

1. Set `DEV_MODE = True`.
2. Run the script and confirm the full pipeline finishes.
3. Set `DEV_MODE = False`.
4. Run full training.

## Main Hyperparameters

| Parameter | Current full value | DEV override |
|-----------|--------------------|--------------|
| `BASELINE_EPOCHS` | 15 | 1 |
| `EXIT_EPOCHS` | 5 | 1 |
| `BASELINE_BATCH` | 4 | 2 |
| `EXIT_BATCH` | 4 | 2 |
| `BASELINE_LR` | `1e-4` | same |
| `EXIT_LR` | `1e-4` | same |
| `EXIT_WARMUP_STEPS` | 1000 | 10 |
| `MAX_LENGTH` | 32 | same |
| `CAPTIONS_PER_IMAGE_TRAIN` | 3 | same |
| `CAPTIONS_PER_IMAGE_VAL` | 3 | same |
| `LAYERS_FOR_EXIT` | `0..11` | same |
| `EXIT_THRESHOLD` | 1.5 | same |
| `EARLY_STOP_PATIENCE` | 3 | 0 |

## Outputs

Each run writes artifacts to:

```text
results/<dev|full>/<timestamp>/
```

Files:

| File | Purpose |
|------|---------|
| `result.log` | Timestamped run log |
| `baseline_step_log.csv` | Per-step baseline losses |
| `baseline_epoch_log.csv` | Per-epoch baseline train/valid losses |
| `baseline_test_metrics.csv` | Baseline BLEU/CIDEr/METEOR |
| `baseline_test_predictions.csv` | Baseline predictions and references |
| `exit_step_log.csv` | Per-step exit-head losses |
| `exit_epoch_log.csv` | Per-epoch exit-head train/valid losses |
| `exit_test_metrics.csv` | Early-exit BLEU/CIDEr/METEOR |
| `exit_test_predictions.csv` | Early-exit predictions and references |
| `exit_layer_usage.csv` | Exit layer histogram |
| `inference_timing.csv` | Baseline vs early-exit latency and ms/token |
| `baseline_train_loss.png` | Baseline train-loss curve |
| `baseline_valid_loss.png` | Baseline valid-loss curve |
| `exit_train_loss.png` | Exit-head train-loss curve |
| `exit_valid_loss.png` | Exit-head valid-loss curve |
| `README.md` (inside result folder) | Run summary |

## Notes and Troubleshooting

### 1) `FileNotFoundError: annotations/captions_train2014.json`

- Ensure `annotations/captions_train2014.json` and
  `annotations/captions_val2017.json` exist in project root.
- If you run from another working directory, use absolute paths in `imgcap.py`.

### 2) METEOR errors

Install Java:

```bash
sudo apt-get install -y default-jre
```

### 3) CUDA OOM

Reduce:

- `BASELINE_BATCH`
- `EXIT_BATCH`
- `MAX_LENGTH`

### 4) Early-exit speedup lower than expected

Tune:

- `EXIT_THRESHOLD`
- `LAYERS_FOR_EXIT`
- image resolution / processor settings

A higher threshold usually improves quality but exits later.
