# Image Captioning with Early Exit (CapEEN)

This README matches the current code in `imgcap.py` and the dependency file
`requirements_jetson.txt`.

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

## Install Dependencies

Install with:

```bash
pip install -r requirements_jetson.txt
sudo apt-get install -y default-jre
```

`requirements_jetson.txt` currently includes:

```text
torch==2.11.0+cu126
torchvision==0.26.0+cu126
datasets==2.21.0
transformers==4.44.2
accelerate==0.34.2
numpy==1.26.4
pillow==10.4.0
tqdm==4.66.5
requests==2.32.3
pycocotools==2.0.8
nltk==3.9.1
sentencepiece==0.2.0
scipy==1.13.1
matplotlib==3.9.2
git+https://github.com/salaniz/pycocoevalcap.git
```

Notes:

- `pycocoevalcap` is already in `requirements_jetson.txt`; do not install it a second time unless the first install fails.
- METEOR requires Java, hence `default-jre`.
- On Jetson, if the official PyTorch CUDA wheel is not available for your Python/aarch64 environment, install the NVIDIA Jetson PyTorch/torchvision wheels manually, then remove or comment the two `torch` lines in `requirements_jetson.txt` before installing the remaining packages.

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
| `baseline_step_log.csv` | Baseline step loss |
| `baseline_epoch_log.csv` | Baseline epoch train/valid loss |
| `baseline_test_metrics.csv` | Baseline BLEU/CIDEr/METEOR |
| `baseline_test_predictions.csv` | Baseline predictions and references |
| `exit_step_log.csv` | Exit-head step loss |
| `exit_epoch_log.csv` | Exit-head epoch train/valid loss |
| `exit_test_metrics.csv` | Early-exit BLEU/CIDEr/METEOR |
| `exit_test_predictions.csv` | Early-exit predictions and references |
| `exit_layer_usage.csv` | Exit-layer frequency |
| `inference_timing.csv` | Baseline vs early-exit latency per image |
| `baseline_train_loss.png` | Baseline train-loss chart |
| `baseline_valid_loss.png` | Baseline valid-loss chart |
| `exit_train_loss.png` | Exit-head train-loss chart |
| `exit_valid_loss.png` | Exit-head valid-loss chart |
| `README.md` | Auto-generated run summary |

Dataset cache files are config-tagged, for example:

```text
train_ds_coco_<dataset_root>_dev1_seed42_val0p9_traincap3_valcap3.pkl
val_ds_coco_<dataset_root>_dev1_seed42_val0p9_traincap3_valcap3.pkl
```

This prevents DEV and FULL dataset caches from being mixed.

## Metrics

The script computes:

| Metric | Meaning |
|--------|---------|
| BLEU-1..4 | n-gram overlap with reference captions |
| CIDEr | Caption relevance/consensus metric |
| METEOR | Precision/recall-oriented caption metric; requires Java |

## Timing / Speedup

The script measures inference latency for each test image:

| Column | Meaning |
|--------|---------|
| `model` | `baseline` or `early_exit` |
| `id` | Test sample id |
| `latency_ms` | Wall-clock inference time in milliseconds |
| `tokens` | Number of generated tokens |
| `ms_per_token` | `latency_ms / tokens` |
| `avg_exit_layer` | Mean exit layer for that image; early-exit only |

CUDA timing is synchronized before and after inference when CUDA is available.

## Current Caveats

- The early-exit decoder uses internal GPT-2 block APIs from Hugging Face Transformers. It is tied to the currently intended Transformers version in `requirements_jetson.txt`.
- Early-exit inference recomputes the generated prefix instead of using KV cache. This is slower than cached decoding, but it avoids inconsistent deep-layer caches when a previous token exits early.
- The current code uses mixed COCO versions: train captions/images from 2014 and validation/test captions/images from 2017. This README documents that behavior exactly.
- Full training on Jetson AGX Orin can take a long time. Use `DEV_MODE=True` first.
