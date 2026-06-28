# =============================================================================
# Image Captioning with Early Exit (CapEEN)
# Encoder : Swin-Base  |  Decoder : GPT-2  |  Dataset : COCO train2014 + val2017
# Hardware : NVIDIA Jetson AGX Orin 64 GB  |  CUDA 12.6 / JetPack 6
# =============================================================================
#
# >>>  BƯỚC ĐẦU TIÊN: chỉnh 3 dòng PATH và DEV_MODE bên dưới  <<<
#
#  Nếu data để trên USB / ổ ngoài, ví dụ:
#    DATASET_ROOT = "/media/usb/coco"          # Linux mount point USB
#    DATASET_ROOT = "/mnt/ssd/coco"            # SSD ngoài
#
#  DEV_MODE = True   → chạy pipeline nhanh trong ~15 phút (debug / kiểm tra)
#  DEV_MODE = False  → chạy full train cho báo cáo
# =============================================================================

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    GPT2TokenizerFast,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
import re
import csv
import json
import pickle
import io
import time
import requests
import urllib.parse as parse
from datetime import datetime
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")          # không cần display server (Jetson headless)


# --- ĐẶT PATH Ở ĐÂY ---
BASELINE_CKPT_BASE = "./image-captioning"  # nơi lưu best baseline model
EXIT_CKPT_DIR_BASE = "./checkpoint/intermediate_head_weights"

ANNOTATIONS_DIR = "annotations"
DATASET_ROOT = "/mnt/usb/coco2014"
TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "train2014")
VAL_IMAGE_DIR = os.path.join(DATASET_ROOT, "val2017")

# --- DEV MODE ---
DEV_MODE = False   # True = pipeline test ~15 phút | False = full train
RUN_MODE = "dev" if DEV_MODE else "full"
BASELINE_CKPT = f"{BASELINE_CKPT_BASE}-{RUN_MODE}"
EXIT_CKPT_DIR = f"{EXIT_CKPT_DIR_BASE}_{RUN_MODE}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device used is", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

transformers.logging.set_verbosity_error()

# =============================================================================
# CONFIG – hyperparameter (không cần chỉnh nếu không có lý do)
# =============================================================================

# --- Models ---
ENCODER_MODEL = "microsoft/swin-base-patch4-window7-224-in22k"
DECODER_MODEL = "gpt2"           # thay "gpt2-medium" nếu muốn decoder lớn hơn

# --- Caption ---
MAX_LENGTH = 32      # số token tối đa mỗi caption
VOCAB_SIZE = 50257   # vocab size GPT-2 (không đổi)
CAPTIONS_PER_IMAGE_TRAIN = 3
CAPTIONS_PER_IMAGE_VAL = 3

# --- Full-run data budget (0 = dùng toàn bộ) ---
FULL_TRAIN_IMAGE_LIMIT = 50000

# --- Val/Test split ---
VAL_SPLIT_RATIO = 0.9   # 90% val2017 -> val, 10% -> test
SPLIT_SEED = 42         # reproducible split

# --- Cache ---
USE_DATASET_CACHE = True
REBUILD_DATASET_CACHE = False

# --- DataLoader ---
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

# --- Logging ---
LOG_EVERY_STEPS = 20   # in log mỗi N step

# --- Phase 1: Baseline fine-tune ---
#   Full train: 5 epoch, batch 4 (Jetson 64 GB unified RAM + AMP đủ)
#   batch=4 an toàn vì Swin-Base encoder ~88M params, GPT-2 ~117M params;
#   peak VRAM ≈ 6–8 GB với AMP fp16, còn xa ngưỡng 64 GB.
BASELINE_EPOCHS = 5
BASELINE_BATCH = 4
BASELINE_LR = 5e-5

# --- Phase 2: Early Exit heads (Knowledge Distillation) ---
#   Chỉ train 12 linear heads nhỏ, best_model frozen → batch có thể cao hơn
EXIT_EPOCHS = 2
EXIT_BATCH = 4
EXIT_LR = 1e-4
EXIT_WARMUP_STEPS = 1000
LAYERS_FOR_EXIT = list(range(12))   # layer 0 → 11 của GPT-2 decoder
EXIT_THRESHOLD = 1.5               # ngưỡng aggregated confidence để exit sớm

# --- Early Stopping ---
# Dừng train sớm nếu valid_loss không cải thiện sau N epoch liên tiếp.
# Tiết kiệm thời gian đáng kể: thực tế COCO thường hội tụ trước epoch 15.
# Tắt bằng cách đặt EARLY_STOP_PATIENCE = 0
EARLY_STOP_PATIENCE = 2  # Phase 1 và Phase 2 dùng chung giá trị này

# Resume: nếu checkpoint baseline đã có thì bỏ qua train baseline để tiết kiệm thời gian
RESUME_BASELINE_FROM_CKPT = True

# Skip baseline test inference/metrics khi đang resume từ checkpoint đã có.
# Dùng khi baseline đã đánh giá xong ở run trước và muốn đi thẳng sang Early Exit.
SKIP_BASELINE_EVAL_WHEN_RESUMED = False

# ---------------------------------------------------------------------------
# DEV MODE – ghi đè tham số để test pipeline trong ~15 phút
# Bật bằng cách đặt DEV_MODE = True ở đầu file
# ---------------------------------------------------------------------------
if DEV_MODE:
    BASELINE_EPOCHS = 1
    EXIT_EPOCHS = 1
    BASELINE_BATCH = 2
    EXIT_BATCH = 2
    EXIT_WARMUP_STEPS = 10
    LOG_EVERY_STEPS = 5
    EARLY_STOP_PATIENCE = 0   # DEV mode không cần early stop
    print("[DEV MODE] Pipeline test: 1 epoch, 200 train / 50 val / 30 test samples")
else:
    print(
        f"[FULL MODE] Train image limit: {FULL_TRAIN_IMAGE_LIMIT if FULL_TRAIN_IMAGE_LIMIT > 0 else 'all'}")

random.seed(SPLIT_SEED)
np.random.seed(SPLIT_SEED)
torch.manual_seed(SPLIT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SPLIT_SEED)

# =============================================================================

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_START = time.perf_counter()
RESULTS_DIR = os.path.join("results", RUN_MODE, RUN_TS)
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(RESULTS_DIR, "result.log")


def log_message(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_csv_row(file_path, fieldnames, row):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[Info] Appended row to {file_path}.")

def save_dict_to_csv(file_path, rows):
    if not rows:
        print(f"[Warning] No rows to save to {file_path}. Skipping CSV write.")
        return
    fieldnames = list(rows[0].keys())
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Info] Saved {len(rows)} rows to {file_path}.")


def save_training_plot(epoch_rows, value_key, out_path, title):
    if not epoch_rows:
        return
    x = [r["epoch"] for r in epoch_rows]
    y = [r[value_key] for r in epoch_rows]
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(value_key)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def sync_cuda_if_needed():
    if device.type == "cuda":
        torch.cuda.synchronize()


def format_seconds(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--:--"
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_readme(summary_path, lines):
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# SECTION 1 – Đọc annotation COCO và xây dựng sample list
# =============================================================================

train_ann_file = os.path.join(ANNOTATIONS_DIR, "captions_train2014.json")
val_ann_file = os.path.join(ANNOTATIONS_DIR, "captions_val2017.json")

with open(train_ann_file, "r") as f:
    train_coco = json.load(f)
with open(val_ann_file, "r") as f:
    val_coco = json.load(f)

# Map image_id -> file_name
train_id2file = {img["id"]: img["file_name"] for img in train_coco["images"]}
val_id2file = {img["id"]: img["file_name"] for img in val_coco["images"]}

# Map image_id -> danh sách caption (mỗi ảnh có 5 caption tham chiếu)
train_image2sentences = {}
for ann in train_coco["annotations"]:
    train_image2sentences.setdefault(
        ann["image_id"], []).append({"raw": ann["caption"]})

if not DEV_MODE and FULL_TRAIN_IMAGE_LIMIT > 0:
    train_ids = list(train_image2sentences.keys())
    rng_train = random.Random(SPLIT_SEED)
    rng_train.shuffle(train_ids)
    selected_ids = set(train_ids[:FULL_TRAIN_IMAGE_LIMIT])
    train_image2sentences = {
        k: v for k, v in train_image2sentences.items() if k in selected_ids}

val_image2sentences = {}
for ann in val_coco["annotations"]:
    val_image2sentences.setdefault(
        ann["image_id"], []).append({"raw": ann["caption"]})

# Tạo train_samples: mỗi cặp (image_path, 1 caption) là 1 sample
train_samples = []
for image_id, sentences in tqdm(train_image2sentences.items(), desc="Build train samples"):
    for cap in sentences[:CAPTIONS_PER_IMAGE_TRAIN]:
        train_samples.append({
            "image_path": os.path.join(TRAIN_IMAGE_DIR, train_id2file[image_id]),
            "caption": cap["raw"],
        })

# Tạo val_samples và test_samples từ val2017:
# - 90% image_id đầu làm val (mỗi caption là 1 sample riêng)
# - 10% cuối làm test (giữ ngựyên 5 caption / ảnh để đánh giá metric đa tham chiếu)
val_samples = []
test_samples = []
rng = random.Random(SPLIT_SEED)
val_ids = list(val_image2sentences.keys())
rng.shuffle(val_ids)
val_cutoff = int(VAL_SPLIT_RATIO * len(val_ids))
for idx, image_id in enumerate(tqdm(val_ids, desc="Build val/test samples")):
    sentences = val_image2sentences[image_id]
    img_path = os.path.join(VAL_IMAGE_DIR, val_id2file[image_id])
    if idx < val_cutoff:
        for cap in sentences[:CAPTIONS_PER_IMAGE_VAL]:
            val_samples.append({"image_path": img_path, "caption": cap["raw"]})
    else:
        test_samples.append({
            "image_path": img_path,
            "caption": [cap["raw"] for cap in sentences],  # giữ cả 5 caption
        })


# DEV MODE: cắt dataset xuống còn nhỏ để test pipeline ~15 phút
if DEV_MODE:
    train_samples = train_samples[:200]
    val_samples = val_samples[:50]
    test_samples = test_samples[:30]

train_unique_images = len({x["image_path"] for x in train_samples})
val_unique_images = len({x["image_path"] for x in val_samples})
test_unique_images = len({x["image_path"] for x in test_samples})
test_ref_count = sum(len(x["caption"]) if isinstance(
    x["caption"], list) else 1 for x in test_samples)
log_message(
    "[Data] "
    f"train_images={train_unique_images}, train_samples={len(train_samples)}, "
    f"val_images={val_unique_images}, val_samples={len(val_samples)}, "
    f"test_images={test_unique_images}, test_samples={len(test_samples)}, test_refs={test_ref_count}"
)

# Cache dataset ra file pkl để lần sau khởi động nhanh hơn (bỏ qua bước xử lý JSON lại)
dataset_root_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                          os.path.abspath(DATASET_ROOT)).strip("_")
split_tag = str(VAL_SPLIT_RATIO).replace(".", "p")
cache_tag = (
    f"{dataset_root_tag}_dev{int(DEV_MODE)}_seed{SPLIT_SEED}_val{split_tag}"
    f"_traincap{CAPTIONS_PER_IMAGE_TRAIN}_valcap{CAPTIONS_PER_IMAGE_VAL}"
)
train_cache_path = f"train_ds_coco_{cache_tag}.pkl"
val_cache_path = f"val_ds_coco_{cache_tag}.pkl"

cache_available = os.path.exists(
    train_cache_path) and os.path.exists(val_cache_path)
if USE_DATASET_CACHE and cache_available and not REBUILD_DATASET_CACHE:
    with open(train_cache_path, "rb") as f:
        train_ds_coco = pickle.load(f)
    with open(val_cache_path, "rb") as f:
        val_ds_coco = pickle.load(f)
else:
    train_ds_coco = Dataset.from_list(train_samples)
    val_ds_coco = Dataset.from_list(val_samples)
    with open(train_cache_path, "wb") as f:
        pickle.dump(train_ds_coco, f)
    with open(val_cache_path, "wb") as f:
        pickle.dump(val_ds_coco, f)

# =============================================================================
# SECTION 2 – Tiện ích đọc ảnh (hỗ trợ cả file cục bộ lẫn URL)
# =============================================================================


def is_url(string: str) -> bool:
    """Kiểm tra chuỗi có phải URL hợp lệ không."""
    try:
        r = parse.urlparse(string)
        return all([r.scheme, r.netloc, r.path])
    except Exception:
        return False


def load_image(image_path: str) -> Image.Image:
    """Load ảnh RGB từ đường dẫn cục bộ hoặc URL."""
    if is_url(image_path):
        response = requests.get(image_path, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return Image.open(image_path).convert("RGB")


# =============================================================================
# SECTION 3 – Load model, tokenizer, image processor
# =============================================================================

# Encoder: Swin-Base pretrained ImageNet-22k → trích xuất đặc trưng ảnh
# Decoder: GPT-2 pretrained text → sinh caption từng token (cross-attention)
decoder_config = AutoConfig.from_pretrained(DECODER_MODEL)
decoder_config.is_decoder = True          # bật chế độ decoder (causal)
decoder_config.add_cross_attention = True  # cho phép attend vào encoder output
decoder_num_layers = decoder_config.n_layer

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    ENCODER_MODEL,
    DECODER_MODEL,
    decoder_config=decoder_config,
    output_hidden_states=True,
).to(device)

tokenizer = GPT2TokenizerFast.from_pretrained(DECODER_MODEL)
image_processor = AutoImageProcessor.from_pretrained(ENCODER_MODEL)

# Cấu hình token đặc biệt cho GPT-2
if "gpt2" in DECODER_MODEL:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

# Load dataset từ biến cache đã tạo/đọc ở Section 1
train_ds = train_ds_coco
valid_ds = val_ds_coco

# Xử lý thành dạng 'pixel_values'-'labels(ids)'


def preprocess(items):
    # preprocess the image
    images = []
    captions = []

    imgs = items["image_path"] if isinstance(items["image_path"], list) else [
        items["image_path"]]
    sentences = items["caption"] if isinstance(
        items["caption"], list) else [items["caption"]]

    for img, sents in zip(imgs, sentences):
        images.append(load_image(img))
        captions.append(sents)

    pixel_values = image_processor(images, return_tensors="pt").pixel_values
    # tokenize the caption with truncation and padding
    targets = tokenizer(
        captions,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values.squeeze(0), "labels": labels.squeeze(0)}


train_dataset = train_ds.with_transform(preprocess)
valid_dataset = valid_ds.with_transform(preprocess)


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }


# DataLoader – dùng tham số từ CONFIG block
train_loader = DataLoader(
    train_dataset,
    collate_fn=collate_fn,
    batch_size=BASELINE_BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
valid_loader = DataLoader(
    valid_dataset,
    collate_fn=collate_fn,
    batch_size=BASELINE_BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# Dataloader riêng cho Phase 2 để EXIT_BATCH có hiệu lực
train_loader_exit = DataLoader(
    train_dataset,
    collate_fn=collate_fn,
    batch_size=EXIT_BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
valid_loader_exit = DataLoader(
    valid_dataset,
    collate_fn=collate_fn,
    batch_size=EXIT_BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# Optimizer và AMP scaler cho Phase 1
current_step = 0
optimizer = AdamW(model.parameters(), lr=BASELINE_LR)
scaler = GradScaler(enabled=device.type == "cuda")

# =============================================================================
# SECTION 5 – Hàm tính metric: BLEU-1~4 / CIDEr / METEOR
# =============================================================================


def compute_caption_metrics(
    predictions: dict[int, list[str]],
    ground_truths: dict[int, list[str]],
    verbose: bool = True
) -> dict[str, float]:
    """
    Tính BLEU-1~4, CIDEr, METEOR cho image captioning.

    Args:
        predictions  : {image_id: ["predicted caption"]}
        ground_truths: {image_id: ["cap1", "cap2", "cap3", "cap4", "cap5"]}
        verbose      : In kết quả ra màn hình

    Returns:
        dict chứa tất cả scores
    """
    assert set(predictions.keys()) == set(ground_truths.keys()), \
        "predictions và ground_truths phải có cùng set image_id"

    scores = {}

    # ---------- BLEU 1~4 ----------
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(ground_truths, predictions)
    for i, score in enumerate(bleu_scores):
        scores[f"BLEU-{i+1}"] = round(score, 4)

    # ---------- CIDEr ----------
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(ground_truths, predictions)
    scores["CIDEr"] = round(cider_score, 4)

    # ---------- METEOR ----------
    try:
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(ground_truths, predictions)
        scores["METEOR"] = round(meteor_score, 4)
    except Exception as exc:
        scores["METEOR"] = float("nan")
        log_message(f"[Metric] METEOR failed: {exc}")

    if verbose:
        print("=" * 35)
        print(f"  {'Metric':<12} {'Score':>10}")
        print("=" * 35)
        for name, val in scores.items():
            print(f"  {name:<12} {val:>10.4f}")
        print("=" * 35)

    return scores


def get_decoder_layer_states(decoder_hidden_states, expected_layers):
    """Trả về list hidden states theo từng decoder layer thực (bỏ embedding state nếu có)."""
    if len(decoder_hidden_states) == expected_layers + 1:
        return list(decoder_hidden_states[1:])
    if len(decoder_hidden_states) == expected_layers:
        return list(decoder_hidden_states)
    return list(decoder_hidden_states[-expected_layers:])


def greedy_decode_true_early_exit(model_ved, inter_heads, pixel_values, start_token_id, eos_token_id, max_length, threshold):
    """
    Early exit thật: chạy decoder layer-by-layer và dừng ngay khi đạt ngưỡng.
    Không chạy full decoder khi đã đủ tự tin.
    """
    transformer = model_ved.decoder.transformer
    inter_heads = list(inter_heads)
    num_layers = len(inter_heads)

    encoder_outputs = model_ved.encoder(
        pixel_values=pixel_values, return_dict=True)
    encoder_hidden_states = encoder_outputs.last_hidden_state

    # Align encoder hidden size to decoder hidden size when they differ (e.g., Swin-Base 1024 -> GPT-2 768).
    if hasattr(model_ved, "enc_to_dec_proj") and model_ved.enc_to_dec_proj is not None:
        encoder_hidden_states = model_ved.enc_to_dec_proj(
            encoder_hidden_states)

    encoder_attention_mask = torch.ones(
        encoder_hidden_states.shape[:2],
        dtype=torch.long,
        device=encoder_hidden_states.device,
    )
    encoder_attention_mask = transformer.invert_attention_mask(
        encoder_attention_mask)

    generated = [start_token_id]
    layer_trace = []

    for _ in range(max_length):
        input_ids = torch.tensor([generated], device=pixel_values.device)
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long,
                               device=pixel_values.device).unsqueeze(0)

        dec_mask = torch.ones((1, seq_len), dtype=torch.long,
                              device=pixel_values.device)
        dec_mask = dec_mask[:, None, None, :].to(
            dtype=transformer.wte.weight.dtype)
        dec_mask = (1.0 - dec_mask) * \
            torch.finfo(transformer.wte.weight.dtype).min

        hidden_states = transformer.wte(input_ids) + transformer.wpe(pos_ids)
        hidden_states = transformer.drop(hidden_states)

        prev_token = None
        agg_conf = 0.0
        chosen_token = None
        chosen_layer = None

        for layer_idx in range(num_layers):
            block = transformer.h[layer_idx]
            block_outputs = block(
                hidden_states,
                layer_past=None,
                attention_mask=dec_mask,
                head_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = block_outputs[0]

            head_hidden_states = transformer.ln_f(
                hidden_states) if layer_idx == num_layers - 1 else hidden_states
            logits = inter_heads[layer_idx](head_hidden_states)
            token = logits[:, -1, :].argmax(-1).item()
            conf = torch.softmax(
                logits[:, -1, :], dim=-1).max(-1).values.item()

            if prev_token is None:
                agg_conf = conf
            elif prev_token == token:
                agg_conf += conf
            else:
                agg_conf = conf

            prev_token = token
            if agg_conf >= threshold:
                chosen_token = token
                chosen_layer = layer_idx
                break

        # Không đủ ngưỡng: fallback về final LM head của baseline.
        if chosen_token is None:
            final_hidden = transformer.ln_f(hidden_states)
            final_logits = model_ved.decoder.lm_head(final_hidden)
            chosen_token = final_logits[:, -1, :].argmax(-1).item()
            chosen_layer = num_layers

        generated.append(chosen_token)
        layer_trace.append(chosen_layer)

        if chosen_token == eos_token_id:
            break

    return generated[1:], layer_trace


# =============================================================================
# SECTION 6 – CSV paths đầu ra (tất cả vào RESULTS_DIR)
# =============================================================================

log_message(f"Run started. Results dir: {RESULTS_DIR}")
log_message(
    f"Run mode: {RUN_MODE}. Baseline checkpoint: {BASELINE_CKPT}. Exit checkpoint dir: {EXIT_CKPT_DIR}")

baseline_step_csv = os.path.join(RESULTS_DIR, "baseline_step_log.csv")
baseline_epoch_csv = os.path.join(RESULTS_DIR, "baseline_epoch_log.csv")
baseline_metrics_csv = os.path.join(RESULTS_DIR, "baseline_test_metrics.csv")
baseline_pred_csv = os.path.join(RESULTS_DIR, "baseline_test_predictions.csv")

exit_step_csv = os.path.join(RESULTS_DIR, "exit_step_log.csv")
exit_epoch_csv = os.path.join(RESULTS_DIR, "exit_epoch_log.csv")
exit_metrics_csv = os.path.join(RESULTS_DIR, "exit_test_metrics.csv")
exit_pred_csv = os.path.join(RESULTS_DIR, "exit_test_predictions.csv")
exit_layer_usage_csv = os.path.join(RESULTS_DIR, "exit_layer_usage.csv")
inference_timing_csv = os.path.join(RESULTS_DIR, "inference_timing.csv")

# =============================================================================
# SECTION 7 – Phase 1: Fine-tune baseline (Swin-Base + GPT-2 end-to-end)
# =============================================================================

best_valid_loss = float("inf")
es_counter_base = 0   # đếm epoch liên tiếp không cải thiện
baseline_ckpt_ready = os.path.exists(os.path.join(BASELINE_CKPT, "config.json"))
baseline_resumed = RESUME_BASELINE_FROM_CKPT and baseline_ckpt_ready
if baseline_resumed:
    log_message(f"[Baseline] Reusing existing checkpoint at {BASELINE_CKPT}, skip baseline training.")
    best_valid_loss = float("nan")
    BASELINE_EPOCHS = 0

for epoch in range(1, BASELINE_EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Baseline Train E{epoch}"), start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            outputs = model(pixel_values=pixel_values,
                            labels=labels, output_hidden_states=True)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step_loss = float(loss.item())
        train_loss_sum += step_loss
        current_step += 1

        append_csv_row(
            baseline_step_csv,
            ["global_step", "epoch", "batch", "step_loss"],
            {
                "global_step": current_step,
                "epoch": epoch,
                "batch": batch_idx,
                "step_loss": round(step_loss, 6),
            },
        )

        if current_step % LOG_EVERY_STEPS == 0:
            log_message(
                f"[Baseline] step={current_step}, epoch={epoch}, batch={batch_idx}, step_loss={step_loss:.6f}")

    train_loss_epoch = train_loss_sum / max(1, len(train_loader))

    # --- Validation ---
    model.eval()
    valid_loss_sum = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Baseline Valid E{epoch}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            label_ids = batch["labels"].to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(pixel_values=pixel_values,
                                labels=label_ids, output_hidden_states=True)
                valid_loss_sum += float(outputs.loss.item())

    valid_loss_epoch = valid_loss_sum / max(1, len(valid_loader))

    append_csv_row(
        baseline_epoch_csv,
        ["epoch", "train_loss", "valid_loss",
            "best_valid_loss_so_far", "saved_checkpoint"],
        {
            "epoch": epoch,
            "train_loss": round(train_loss_epoch, 6),
            "valid_loss": round(valid_loss_epoch, 6),
            "best_valid_loss_so_far": round(min(best_valid_loss, valid_loss_epoch), 6),
            "saved_checkpoint": int(valid_loss_epoch < best_valid_loss),
        },
    )

    log_message(
        f"[Baseline][Epoch {epoch}] train_loss={train_loss_epoch:.6f}, valid_loss={valid_loss_epoch:.6f}, best_valid_loss={best_valid_loss:.6f}"
    )

    if valid_loss_epoch < best_valid_loss:
        best_valid_loss = valid_loss_epoch
        es_counter_base = 0   # reset khi có cải thiện
        model.save_pretrained(BASELINE_CKPT)
        tokenizer.save_pretrained(BASELINE_CKPT)
        image_processor.save_pretrained(BASELINE_CKPT)
        log_message(
            f"[Baseline] New best model saved  valid_loss={valid_loss_epoch:.6f}")
    else:
        es_counter_base += 1
        log_message(
            f"[Baseline] No improvement {es_counter_base}/{EARLY_STOP_PATIENCE}")
        if EARLY_STOP_PATIENCE > 0 and es_counter_base >= EARLY_STOP_PATIENCE:
            log_message(
                f"[Baseline] Early stopping triggered at epoch {epoch}")
            break


# =============================================================================
# SECTION 8 – Evaluate baseline trên test set (greedy decode)
# =============================================================================

preds = {}
gts = {sample_id: item["caption"] for sample_id, item in enumerate(test_samples)}
baseline_pred_rows = []
baseline_timing_rows = []

if baseline_resumed and SKIP_BASELINE_EVAL_WHEN_RESUMED:
    log_message("[Baseline] Skip baseline test inference/metrics (resumed checkpoint).")
    scores = {
        "BLEU-1": float("nan"),
        "BLEU-2": float("nan"),
        "BLEU-3": float("nan"),
        "BLEU-4": float("nan"),
        "CIDEr": float("nan"),
        "METEOR": float("nan"),
    }
else:
    baseline_eval_model = VisionEncoderDecoderModel.from_pretrained(
        BASELINE_CKPT).to(device)
    baseline_eval_model.eval()
    baseline_eval_tokenizer = AutoTokenizer.from_pretrained(BASELINE_CKPT)
    baseline_eval_processor = AutoImageProcessor.from_pretrained(BASELINE_CKPT)

    for sample_id, item in enumerate(tqdm(test_samples, desc="Baseline Test Inference")):
        image = load_image(item["image_path"])
        pixel = baseline_eval_processor(
            image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            sync_cuda_if_needed()
            infer_start = time.perf_counter()
            predict = baseline_eval_model.generate(pixel, max_length=MAX_LENGTH)
            sync_cuda_if_needed()
            latency_ms = (time.perf_counter() - infer_start) * 1000.0
        cap_predict = baseline_eval_tokenizer.batch_decode(
            predict, skip_special_tokens=True)[0]
        cap_predict = re.sub(r"\s+", " ", cap_predict).strip()
        output_tokens = int(predict.shape[-1])

        preds[sample_id] = [cap_predict]

        baseline_pred_rows.append(
            {
                "id": sample_id,
                "image_path": item["image_path"],
                "prediction": cap_predict,
                "ground_truth": " || ".join(item["caption"]) if isinstance(item["caption"], list) else str(item["caption"]),
            }
        )
        baseline_timing_rows.append(
            {
                "model": "baseline",
                "id": sample_id,
                "latency_ms": round(latency_ms, 3),
                "tokens": output_tokens,
                "ms_per_token": round(latency_ms / max(1, output_tokens), 3),
                "avg_exit_layer": "",
            }
        )

    scores = compute_caption_metrics(preds, gts)
    log_message("[Baseline] Final test metrics: " +
                ", ".join([f"{k}={v:.4f}" for k, v in scores.items()]))

    save_dict_to_csv(
        baseline_metrics_csv,
        [{"metric": k, "score": v} for k, v in scores.items()],
    )
    save_dict_to_csv(baseline_pred_csv, baseline_pred_rows)


# =============================================================================
# SECTION 9 – Phase 2: Train Early Exit heads (Knowledge Distillation)
# =============================================================================

# Load best baseline checkpoint để làm teacher, freeze toàn bộ tham số
best_model = VisionEncoderDecoderModel.from_pretrained(
    BASELINE_CKPT).to(device)
best_model.eval()
for p in best_model.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(BASELINE_CKPT)
image_processor = AutoImageProcessor.from_pretrained(BASELINE_CKPT)

num_epochs_exit = EXIT_EPOCHS
current_step_exit = 0


class IntermediateHead(nn.Module):
    """
    Linear head gắn vào hidden state của từng decoder layer.
    Project từ hidden_size → vocab_size để dự đoán token tại mỗi layer trung gian.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


layers_for_exit = LAYERS_FOR_EXIT
intermediate_heads = nn.ModuleList(
    [IntermediateHead(decoder_config.hidden_size, VOCAB_SIZE)
     for _ in layers_for_exit]
).to(device)

optimizer_exit = AdamW(intermediate_heads.parameters(), lr=EXIT_LR)
scheduler_exit = get_linear_schedule_with_warmup(
    optimizer_exit,
    num_warmup_steps=EXIT_WARMUP_STEPS,
    num_training_steps=max(1, EXIT_EPOCHS * len(train_loader_exit)),
)
scaler_heads = GradScaler(enabled=device.type == "cuda")
kl_div_loss = nn.KLDivLoss(reduction="batchmean")

best_exit_valid_loss = float("inf")
es_counter_exit = 0   # đếm số epoch liên tiếp không cải thiện (early stop)
intermediate_head_weights_dir = EXIT_CKPT_DIR
os.makedirs(intermediate_head_weights_dir, exist_ok=True)

for epoch in range(1, num_epochs_exit + 1):
    intermediate_heads.train()
    train_loss_sum = 0.0
    epoch_start_time = time.perf_counter()
    total_batches_exit = max(1, len(train_loader_exit))

    log_message(
        f"[Exit] Epoch {epoch}/{num_epochs_exit} started. total_steps={total_batches_exit}"
    )

    for batch_idx, batch in enumerate(tqdm(train_loader_exit, desc=f"Exit Train E{epoch}"), start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=device.type == "cuda"):
            # best_model frozen, chỉ lấy hidden states làm input cho các head
            outputs = best_model(pixel_values=pixel_values,
                                 labels=labels, output_hidden_states=True)
            layer_states = get_decoder_layer_states(
                outputs.decoder_hidden_states, decoder_num_layers)
            valid_token_mask = labels.ne(-100)
            teacher_logits = outputs.logits.detach()
            int_loss_train = 0.0
            for exit_idx in range(len(layers_for_exit)):
                inter_logits = intermediate_heads[exit_idx](
                    layer_states[layers_for_exit[exit_idx]]
                )
                # KL divergence: học soft distribution của final layer (teacher)
                inter_logits_valid = inter_logits[valid_token_mask]
                teacher_logits_valid = teacher_logits[valid_token_mask]
                kl_loss = kl_div_loss(
                    F.log_softmax(inter_logits_valid, dim=-1),
                    F.softmax(teacher_logits_valid, dim=-1),
                )
                # Cross-entropy: học dự đoán đúng ground-truth token
                ce_loss = F.cross_entropy(
                    inter_logits_valid,
                    labels[valid_token_mask],
                )
                int_loss_train += 0.5 * ce_loss + 0.5 * kl_loss

            int_loss_train = int_loss_train / len(layers_for_exit)

        scaler_heads.scale(int_loss_train).backward()
        scaler_heads.step(optimizer_exit)
        scaler_heads.update()
        scheduler_exit.step()
        optimizer_exit.zero_grad()

        step_loss = float(int_loss_train.item())
        train_loss_sum += step_loss
        current_step_exit += 1

        append_csv_row(
            exit_step_csv,
            ["global_step", "epoch", "batch", "step_loss"],
            {
                "global_step": current_step_exit,
                "epoch": epoch,
                "batch": batch_idx,
                "step_loss": round(step_loss, 6),
            },
        )

        if current_step_exit % LOG_EVERY_STEPS == 0:
            elapsed = time.perf_counter() - epoch_start_time
            step_rate = batch_idx / max(elapsed, 1e-9)
            remaining_steps = max(0, total_batches_exit - batch_idx)
            eta_seconds = remaining_steps / max(step_rate, 1e-9)
            progress_pct = (batch_idx / total_batches_exit) * 100.0
            log_message(
                f"[Exit] epoch={epoch}/{num_epochs_exit}, step={batch_idx}/{total_batches_exit} "
                f"({progress_pct:.1f}%), loss={step_loss:.6f}, eta={format_seconds(eta_seconds)}")

    train_loss_epoch = train_loss_sum / max(1, len(train_loader_exit))

    # --- Validation (weighted CE theo layer index, layer sâu hơn penalty cao hơn) ---
    intermediate_heads.eval()
    valid_loss_sum = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader_exit, desc=f"Exit Valid E{epoch}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            label_ids = batch["labels"].to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = best_model(
                    pixel_values=pixel_values, labels=label_ids, output_hidden_states=True)
                layer_states = get_decoder_layer_states(
                    outputs.decoder_hidden_states, decoder_num_layers)
                int_loss = 0.0
                weight_sum = 0.0
                for exit_idx in range(len(layers_for_exit)):
                    inter_logits = intermediate_heads[exit_idx](
                        layer_states[layers_for_exit[exit_idx]]
                    )
                    # layer sâu hơn được weight cao hơn để phạt exit muộn
                    weight = (layers_for_exit[exit_idx] + 1)
                    int_loss += weight * F.cross_entropy(
                        inter_logits.view(-1, inter_logits.size(-1)),
                        label_ids.view(-1),
                    )
                    weight_sum += weight
                int_loss = int_loss / max(1.0, weight_sum)
            valid_loss_sum += float(int_loss.item())

    valid_loss_epoch = valid_loss_sum / max(1, len(valid_loader_exit))

    is_best_exit = valid_loss_epoch < best_exit_valid_loss
    if is_best_exit:
        best_exit_valid_loss = valid_loss_epoch
        es_counter_exit = 0   # reset khi có cải thiện
        for layer_idx, head in enumerate(intermediate_heads):
            head_path = os.path.join(
                intermediate_head_weights_dir, f"head_layer_{layers_for_exit[layer_idx]}.pt")
            torch.save(head.state_dict(), head_path)
    else:
        es_counter_exit += 1
        log_message(
            f"[Exit] No improvement {es_counter_exit}/{EARLY_STOP_PATIENCE}")
        if EARLY_STOP_PATIENCE > 0 and es_counter_exit >= EARLY_STOP_PATIENCE:
            log_message(f"[Exit] Early stopping triggered at epoch {epoch}")
            break

    append_csv_row(
        exit_epoch_csv,
        ["epoch", "train_loss", "valid_loss",
            "best_valid_loss_so_far", "saved_checkpoint"],
        {
            "epoch": epoch,
            "train_loss": round(train_loss_epoch, 6),
            "valid_loss": round(valid_loss_epoch, 6),
            "best_valid_loss_so_far": round(best_exit_valid_loss, 6),
            "saved_checkpoint": int(is_best_exit),
        },
    )

    log_message(
        f"[Exit][Epoch {epoch}] train_loss={train_loss_epoch:.6f}, valid_loss={valid_loss_epoch:.6f}, best_valid_loss={best_exit_valid_loss:.6f}"
    )


# =============================================================================
# SECTION 10 – Inference Early Exit trên test set
# =============================================================================

# Load best exit head weights
for layer_idx, head in enumerate(intermediate_heads):
    head_path = os.path.join(
        intermediate_head_weights_dir, f"head_layer_{layers_for_exit[layer_idx]}.pt")
    if os.path.exists(head_path):
        state_dict = torch.load(head_path, map_location=device, weights_only=True)
        head.load_state_dict(state_dict)
    head.to(device)
    head.eval()

predictions = {}
layer_list = []
exit_pred_rows = []
exit_timing_rows = []

with torch.no_grad():
    for sample_id, item in enumerate(tqdm(test_samples, desc="Exit Test Inference")):
        image = load_image(item["image_path"])
        pixel_values = image_processor(
            image, return_tensors="pt").pixel_values.to(device)
        start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        sync_cuda_if_needed()
        infer_start = time.perf_counter()
        pred_ids, per_token_layers = greedy_decode_true_early_exit(
            best_model,
            intermediate_heads,
            pixel_values,
            start_token_id=start_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=MAX_LENGTH,
            threshold=EXIT_THRESHOLD,
        )
        sync_cuda_if_needed()
        latency_ms = (time.perf_counter() - infer_start) * 1000.0
        layer_list.extend(per_token_layers)
        output_tokens = len(pred_ids)

        cap_predict = tokenizer.decode(pred_ids, skip_special_tokens=True)
        cap_predict = re.sub(r"\s+", " ", cap_predict).strip()
        predictions[sample_id] = [cap_predict]

        exit_pred_rows.append(
            {
                "id": sample_id,
                "image_path": item["image_path"],
                "prediction": cap_predict,
                "ground_truth": " || ".join(item["caption"]) if isinstance(item["caption"], list) else str(item["caption"]),
            }
        )
        exit_timing_rows.append(
            {
                "model": "early_exit",
                "id": sample_id,
                "latency_ms": round(latency_ms, 3),
                "tokens": output_tokens,
                "ms_per_token": round(latency_ms / max(1, output_tokens), 3),
                "avg_exit_layer": round(float(np.mean(per_token_layers)), 3) if per_token_layers else "",
            }
        )

scores_exit = compute_caption_metrics(predictions, gts)
log_message("[Exit] Final test metrics: " +
            ", ".join([f"{k}={v:.4f}" for k, v in scores_exit.items()]))

save_dict_to_csv(exit_metrics_csv, [
                 {"metric": k, "score": v} for k, v in scores_exit.items()])
save_dict_to_csv(exit_pred_csv, exit_pred_rows)

layer_usage_rows = []
if layer_list:
    unique_layers, counts = np.unique(np.array(layer_list), return_counts=True)
    total = int(np.sum(counts))
    for l, c in zip(unique_layers.tolist(), counts.tolist()):
        layer_usage_rows.append(
            {
                "layer_index": l,
                "count": c,
                "ratio": round(c / max(1, total), 6),
            }
        )
save_dict_to_csv(exit_layer_usage_csv, layer_usage_rows)

all_timing_rows = baseline_timing_rows + exit_timing_rows
save_dict_to_csv(inference_timing_csv, all_timing_rows)

baseline_avg_latency = float(np.mean(
    [r["latency_ms"] for r in baseline_timing_rows])) if baseline_timing_rows else 0.0
exit_avg_latency = float(np.mean(
    [r["latency_ms"] for r in exit_timing_rows])) if exit_timing_rows else 0.0
baseline_avg_ms_per_token = float(np.mean(
    [r["ms_per_token"] for r in baseline_timing_rows])) if baseline_timing_rows else 0.0
exit_avg_ms_per_token = float(np.mean(
    [r["ms_per_token"] for r in exit_timing_rows])) if exit_timing_rows else 0.0
avg_exit_layer = float(np.mean(layer_list)) if layer_list else 0.0
speedup = baseline_avg_latency / exit_avg_latency if exit_avg_latency > 0 else 0.0

log_message(
    "[Timing] "
    f"baseline_avg_latency_ms={baseline_avg_latency:.3f}, "
    f"exit_avg_latency_ms={exit_avg_latency:.3f}, "
    f"speedup={speedup:.3f}x, "
    f"avg_exit_layer={avg_exit_layer:.3f}"
)


# charts
baseline_epoch_rows = []
exit_epoch_rows = []

if os.path.exists(baseline_epoch_csv):
    with open(baseline_epoch_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            baseline_epoch_rows.append(
                {
                    "epoch": int(r["epoch"]),
                    "train_loss": float(r["train_loss"]),
                    "valid_loss": float(r["valid_loss"]),
                }
            )

if os.path.exists(exit_epoch_csv):
    with open(exit_epoch_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            exit_epoch_rows.append(
                {
                    "epoch": int(r["epoch"]),
                    "train_loss": float(r["train_loss"]),
                    "valid_loss": float(r["valid_loss"]),
                }
            )

save_training_plot(
    baseline_epoch_rows,
    "train_loss",
    os.path.join(RESULTS_DIR, "baseline_train_loss.png"),
    "Baseline Train Loss",
)
save_training_plot(
    baseline_epoch_rows,
    "valid_loss",
    os.path.join(RESULTS_DIR, "baseline_valid_loss.png"),
    "Baseline Validation Loss",
)
save_training_plot(
    exit_epoch_rows,
    "train_loss",
    os.path.join(RESULTS_DIR, "exit_train_loss.png"),
    "Early Exit Train Loss",
)
save_training_plot(
    exit_epoch_rows,
    "valid_loss",
    os.path.join(RESULTS_DIR, "exit_valid_loss.png"),
    "Early Exit Validation Loss",
)


# readme report
readme_lines = [
    "# Training Result Summary",
    "",
    f"- Run timestamp: {RUN_TS}",
    f"- Run mode: {RUN_MODE}",
    f"- Device: {device}",
    f"- Baseline checkpoint: {BASELINE_CKPT}",
    f"- Exit checkpoint dir: {EXIT_CKPT_DIR}",
    f"- Baseline best valid loss: {best_valid_loss:.6f}",
    f"- Exit best valid loss: {best_exit_valid_loss:.6f}",
    "",
    "## Baseline Test Metrics",
]
for k, v in scores.items():
    readme_lines.append(f"- {k}: {v:.4f}")

readme_lines.append("")
readme_lines.append("## Early Exit Test Metrics")
for k, v in scores_exit.items():
    readme_lines.append(f"- {k}: {v:.4f}")

readme_lines.extend(
    [
        "",
        "## Inference Timing",
        f"- Baseline avg latency: {baseline_avg_latency:.3f} ms/image",
        f"- Early exit avg latency: {exit_avg_latency:.3f} ms/image",
        f"- Speedup: {speedup:.3f}x",
        f"- Baseline avg ms/token: {baseline_avg_ms_per_token:.3f}",
        f"- Early exit avg ms/token: {exit_avg_ms_per_token:.3f}",
        f"- Early exit avg layer: {avg_exit_layer:.3f}",
    ]
)

readme_lines.extend(
    [
        "",
        "## Output Files",
        "- result.log",
        "- baseline_step_log.csv",
        "- baseline_epoch_log.csv",
        "- baseline_test_metrics.csv",
        "- baseline_test_predictions.csv",
        "- exit_step_log.csv",
        "- exit_epoch_log.csv",
        "- exit_test_metrics.csv",
        "- exit_test_predictions.csv",
        "- exit_layer_usage.csv",
        "- inference_timing.csv",
        "- baseline_train_loss.png",
        "- baseline_valid_loss.png",
        "- exit_train_loss.png",
        "- exit_valid_loss.png",
    ]
)

write_readme(os.path.join(RESULTS_DIR, "README.md"), readme_lines)
log_message("All report artifacts generated successfully.")
total_seconds = time.perf_counter() - RUN_START
hours, rem = divmod(int(total_seconds), 3600)
minutes, seconds = divmod(rem, 60)

log_message(
    f"[Duration] total_runtime={hours:02d}:{minutes:02d}:{seconds:02d} "
    f"({total_seconds:.2f}s)"
)
