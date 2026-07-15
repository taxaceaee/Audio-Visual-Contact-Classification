# Image-Only Pipeline — Best Path (Input → Output)

Hướng dẫn chi tiết **nhánh image-only tốt nhất** trong repo `tree_audio`:  
**ResNet18 fine-tune → TTA + class bias → robot/test**.

| | |
|--|--|
| **Robot Macro F1** | **0.450474** |
| **Robot Accuracy** | 0.488058 |
| **Hand val Macro F1** | 0.576328 |
| **Run slug (final)** | `image_dl_resnet18_tta_bias_specimen_select` |
| **Source fine-tune** | `image_dl_finetune_resnet18_specimen_select` |
| **Selected recipe** | `plain` |
| **Selected bias** | `[0.0, 0.0, -1.2, -1.0]` (ambient, leaf, trunk, twig) |

**Scripts:**

1. `train_image_dl_finetune_select_final_test.py` — fine-tune ResNet18, lock best epoch  
2. `tune_image_dl_checkpoint_bias_tta_select_final_test.py` — TTA + bias trên checkpoint, lock rồi eval robot  

**Shared utils:** `train_image_handcrafted_ml_select_final_test.py` (manifest, split, bias, report)

**Constraint:** chỉ dùng ảnh. Không audio features, không multimodal, không filename làm input model.

---

## Mục lục

1. [Tổng quan luồng](#1-tổng-quan-luồng)
2. [Input: dữ liệu](#2-input-dữ-liệu)
3. [Protocol lock-before-test](#3-protocol-lock-before-test)
4. [Bước 1 — Load manifest & split](#4-bước-1--load-manifest--split)
5. [Bước 2 — Fine-tune ResNet18](#5-bước-2--fine-tune-resnet18)
6. [Bước 3 — TTA + class bias](#6-bước-3--tta--class-bias)
7. [Bước 4 — Robot final test](#7-bước-4--robot-final-test)
8. [Output artifacts](#8-output-artifacts)
9. [Cách chạy](#9-cách-chạy)
10. [Inference 1 ảnh](#10-inference-1-ảnh)
11. [Metrics](#11-metrics)
12. [Giới hạn](#12-giới-hạn)

---

## 1. Tổng quan luồng

```
 HAND dataset.csv + images/
           │
           ▼
 load_image_manifest → image_path, y ∈ {0,1,2,3}, specimen_group
           │
           ▼
 split_train_val (specimen, ~20% val)  ── assert không group leak
           │
           ▼
 ┌─────────────────────────────────────────────────────────┐
 │  PHASE A — Fine-tune                                    │
 │  train: aug + WeightedSampler + weighted CE             │
 │  warmup 2 epoch (freeze backbone) → full train          │
 │  mỗi epoch: val Macro F1 → save best_val_model.pt       │
 │  LOCK #1: selected_without_test.json (best epoch)       │
 └─────────────────────────────────────────────────────────┘
           │
           ▼  checkpoint ResNet18
 ┌─────────────────────────────────────────────────────────┐
 │  PHASE B — TTA + bias (hand val only)                   │
 │  recipes: plain / hflip / center crops                  │
 │  bias: none | tuned (grid on log-proba)                 │
 │  select: regularized_val_score = F1 − λ·‖bias‖₁         │
 │  LOCK #2: recipe + bias (selected_without_test.json)    │
 └─────────────────────────────────────────────────────────┘
           │
           ▼  lần đầu load robot
 ROBOT dataset.csv + images/
           │
           ▼
 TTA(selected recipe) → proba → log(proba)+bias → argmax
           │
           ▼
 pred_label, proba_*, final_test_report.csv
 Macro F1 ≈ 0.4505
```

### Label map

| Label | ID |
|-------|----|
| `ambient` | 0 |
| `leaf` | 1 |
| `trunk` | 2 |
| `twig` | 3 |

---

## 2. Input: dữ liệu

### 2.1 Dataset root

Script tự tìm thư mục chứa:

```
<audio_visual_dataset_default>/dataset.csv
<audio_visual_dataset_robo_default>/dataset.csv
```

Hàm: `resolve_root()` — thử `./tree_structures`, `../tree_base/tree_structures`,  
`/home/ttung05/Desktop/tree_base/tree_structures`, … hoặc `--root`.

### 2.2 Cấu trúc

```
tree_structures/
├── audio_visual_dataset_default/          # HAND — train + val
│   ├── dataset.csv
│   └── images/*.jpg
└── audio_visual_dataset_robo_default/     # ROBOT — final test only
    ├── dataset.csv
    └── images/*.jpg
```

`audio_file` trong CSV chỉ để report/alignment — **không** vào model.

### 2.3 Format `dataset.csv`

```csv
audio_file,image_file,category
audio/letwig1.4_segment_0_window_0_ambient.wav,images/letwig1.4_segment_0_window_0_ambient.jpg,ambient
```

| Cột | Bắt buộc | Vai trò |
|-----|----------|---------|
| `image_file` | Có | Relative path ảnh |
| `category` | Có | `ambient` / `leaf` / `trunk` / `twig` |
| `audio_file` | Không | Chỉ lưu report |

### 2.4 Quy mô thực tế

| Split | Rows | Unique exact images |
|-------|------|---------------------|
| Hand | 10 676 | 235 |
| Robot | 2 219 | 45 |

Nhiều row share cùng pixels nhưng khác label (label phản ánh audio contact state) → ceiling image-only bị siết.

---

## 3. Protocol lock-before-test

```
1. Load hand ONLY
2. Split train/val (specimen)
3. Train / search chỉ trên hand
4. Ghi selected_without_test.json   ◄── LOCK
5. Load robot
6. Infer đúng config đã lock
7. Ghi final report / predictions
```

| Được | Không được |
|------|------------|
| Ảnh RGB | Audio features |
| Bias tune trên **hand val** | Tune bias trên robot |
| Chọn epoch/recipe bằng hand val | Chọn theo robot F1 |
| `audio_file` trong report | Filename làm feature model |

---

## 4. Bước 1 — Load manifest & split

**File utils:** `train_image_handcrafted_ml_select_final_test.py`

### 4.1 `load_image_manifest`

Mỗi row →:

| Cột | Nguồn |
|-----|--------|
| `image_path` | `dataset_dir / image_file` |
| `label` | `category.lower()` |
| `y` | `LABEL_MAP[label]` |
| `segment_group` | stem bỏ `_window_\d+.*` |
| `specimen_group` | stem bỏ `_segment_.*` |
| `source` | `"hand_train"` hoặc `"robot_test"` |

Ví dụ:

```
file:     letwig1.4_segment_0_window_1_ambient.jpg
segment:  letwig1.4_segment_0
specimen: letwig1.4
```

Validation: sau filter phải còn đủ 4 class; thiếu → `ValueError`.

### 4.2 `split_train_val` (specimen)

```python
train_idx, val_idx, split_info = split_train_val(
    frame, val_size=0.2, random_state=42, split_mode="specimen"
)
```

- `StratifiedGroupKFold` theo `specimen_group`
- Thử nhiều seed, chọn fold: đủ 4 class, size ~20%, phân bố class gần full
- **Assert** train groups ∩ val groups = ∅
- Lưu manifests vào `outputs/.../<run_slug>/splits/`

---

## 5. Bước 2 — Fine-tune ResNet18

**Script:** `train_image_dl_finetune_select_final_test.py`  
**Run:** `image_dl_finetune_resnet18_specimen_select`

### 5.1 Dataset

```python
class ImageDataset:
    # open RGB; missing/corrupt → black 640×480
    return transform(image), int(y)
```

### 5.2 Transforms

**Train** (`train_transform`, size=224):

1. `RandomResizedCrop(224, scale=(0.65,1.0), ratio=(0.85,1.2))`
2. `RandomHorizontalFlip(0.5)`
3. `ColorJitter(0.25, 0.25, 0.20, 0.04)` p=0.8
4. `RandomGrayscale(p=0.08)`
5. `GaussianBlur(3, sigma=(0.1,1.2))`
6. `ToTensor` + ImageNet normalize  
   mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`
7. `RandomErasing(p=0.15, scale=(0.02,0.12))`

**Eval** (`eval_transform`):

1. `Resize((224, 224), BICUBIC)`
2. `ToTensor` + ImageNet normalize

### 5.3 Model

```python
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = Sequential(Dropout(0.2), Linear(in_features, 4))
# output logits: (B, 4)
```

### 5.4 Training defaults

| Hyperparam | Value |
|------------|-------|
| epochs | 12 (max) |
| warmup_epochs | 2 (chỉ train head) |
| batch_size | 32 |
| lr_head | 1e-3 |
| lr_backbone | 1e-4 |
| weight_decay | 1e-4 |
| dropout | 0.2 |
| sampler | weighted (1 / class_count) |
| loss | weighted CE (`n / count`, normalize mean=1) |
| optimizer | AdamW, 2 param groups |
| AMP | GradScaler + autocast (CUDA) |
| patience | 5 stale epochs |
| selection metric | hand val `macro_f1_4class` |
| final_train_mode | `best_checkpoint` |

**Warmup:**

```
epoch ≤ 2: freeze backbone, train head only
epoch > 2: unfreeze backbone, differential LR
```

**Mỗi epoch:**

1. `train_one_epoch` → train loss  
2. `evaluate` val → logits → Macro F1  
3. Nếu best → save:

```
outputs/audio_feature_benchmarks/
  image_dl_finetune_resnet18_specimen_select/models/
    image_dl_finetune_resnet18_specimen_select_best_val_model.pt
```

Checkpoint:

```python
{
  "model_state": ...,
  "args": ...,
  "best_row": {"epoch", "macro_f1_4class", ...},
  "label_map": {"ambient": 0, "leaf": 1, "trunk": 2, "twig": 3},
}
```

### 5.5 LOCK #1

Sau khi chọn best epoch (vẫn **chưa** load robot):

```
reports/image_dl_finetune_resnet18_specimen_select_selected_without_test.json
```

Chứa `selected_epoch`, `best_checkpoint`, val metrics.

Fine-tune thuần (argmax logits, không bias) đạt robot Macro F1 ~**0.422**.  
Bước 3 cải thiện thêm bằng TTA search + bias.

---

## 6. Bước 3 — TTA + class bias

**Script:** `tune_image_dl_checkpoint_bias_tta_select_final_test.py`  
**Run:** `image_dl_resnet18_tta_bias_specimen_select`

### 6.1 Load checkpoint

```
source = image_dl_finetune_resnet18_specimen_select
checkpoint = .../models/..._best_val_model.pt
model.load_state_dict(checkpoint["model_state"]); model.eval()
```

Hand lại được split specimen (cùng protocol) → chỉ dùng **val** để search.

### 6.2 TTA variants

| Variant | Pipeline |
|---------|----------|
| `plain` | Resize(224) → Tensor → Norm |
| `hflip` | Resize(224) → mirror → Tensor → Norm |
| `center256` | Resize(256) → CenterCrop(224) → Tensor → Norm |
| `center320` | Resize(320) → CenterCrop(224) → Tensor → Norm |

**Recipes** (trung bình softmax):

| Recipe | Variants |
|--------|----------|
| `plain` | plain |
| `plain_hflip` | plain + hflip |
| `plain_center256` | plain + center256 |
| `plain_hflip_center256` | plain + hflip + center256 |
| `all4` | plain + hflip + center256 + center320 |

```python
proba = mean_i  softmax(model(T_i(image)))   # shape (N, 4)
```

### 6.3 Class bias

```python
# decode
scores = log(clip(proba, 1e-12, 1.0)) + bias   # bias (4,)
pred = argmax(scores)

# tune trên hand val
# 1) grid leaf/trunk/twig: linspace(-1.2, 1.2, 13), ambient=0
# 2) refine ambient: linspace(-0.8, 0.8, 9)
# maximize macro F1
```

Mỗi recipe thử `bias_mode ∈ {none, tuned}`.

### 6.4 Selection metric

```python
regularized_val_score = macro_f1_4class - bias_penalty * sum(|bias|)
# bias_penalty default = 0.02
```

Leaderboard sort theo `regularized_val_score` (rồi macro F1, accuracy).

### 6.5 LOCK #2 (kết quả thực tế)

| Field | Value |
|-------|-------|
| recipe | `plain` |
| bias_mode | `tuned` |
| bias | `[0.0, 0.0, -1.2, -1.0]` |
| hand val Macro F1 | 0.576328 |
| regularized score | 0.532328 |

Ghi:

```
reports/image_dl_resnet18_tta_bias_specimen_select_selected_without_test.json
```

**Sau file này mới được load robot.**

> Ghi chú: multi-view TTA không thắng `plain` trên regularized score;  
> phần lift chính so với fine-tune thuần đến từ **bias tuned** trên log-proba.

---

## 7. Bước 4 — Robot final test

```
1. load_image_manifest(robot)
2. predict_proba(model, robot_df, recipe="plain")
3. pred = predict_with_bias(proba, bias=[0, 0, -1.2, -1.0])
4. metrics → final_test_report.csv
5. confusion matrix, predictions CSV, protocol_summary.json
```

**Kết quả locked:**

| Metric | Value |
|--------|------:|
| Macro F1 4-class | **0.450474** |
| Accuracy 4-class | 0.488058 |
| Contact Macro F1 | 0.419445 |
| Binary Macro F1 | 0.503266 |

---

## 8. Output artifacts

### Phase A (fine-tune)

```
outputs/audio_feature_benchmarks/image_dl_finetune_resnet18_specimen_select/
├── models/
│   └── image_dl_finetune_resnet18_specimen_select_best_val_model.pt
├── splits/
│   ├── train_manifest.csv
│   └── val_manifest.csv
└── reports/
    ├── ..._val_history.csv
    ├── ..._selected_without_test.json      # LOCK #1
    ├── ..._split_summary.json
    ├── ..._final_test_report.csv           # fine-tune only (~0.422)
    ├── ..._final_test_predictions.csv
    └── ..._protocol_summary.json
```

### Phase B (TTA + bias) — artifacts claim

```
outputs/audio_feature_benchmarks/image_dl_resnet18_tta_bias_specimen_select/
├── splits/
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   └── test_manifest.csv
└── reports/
    ├── ..._val_leaderboard.csv
    ├── ..._selected_without_test.json      # LOCK #2 ★
    ├── ..._split_summary.json
    ├── ..._final_test_report.csv           # ★ 0.450474
    ├── ..._final_test_confusion_matrix.csv
    ├── ..._final_test_predictions.csv
    └── ..._protocol_summary.json
```

### Predictions CSV (mỗi robot row)

| Cột | Ý nghĩa |
|-----|---------|
| `image_file`, `image_path` | Identity |
| `label`, `y` | Ground truth |
| `pred_y`, `pred_label` | Prediction |
| `proba_ambient` … `proba_twig` | Softmax (TTA-averaged) |
| `segment_group`, `specimen_group` | Group keys |

---

## 9. Cách chạy

Từ root `tree_audio` (GPU khuyến nghị):

### Phase A — fine-tune

```bash
python train_image_dl_finetune_select_final_test.py \
  --run-slug image_dl_finetune_resnet18_specimen_select \
  --model resnet18 \
  --split-mode specimen \
  --val-size 0.2 \
  --random-state 42 \
  --epochs 12 \
  --warmup-epochs 2 \
  --batch-size 32 \
  --image-size 224 \
  --sampler weighted \
  --loss weighted_ce \
  --final-train-mode best_checkpoint
```

### Phase B — TTA + bias + robot

```bash
python tune_image_dl_checkpoint_bias_tta_select_final_test.py \
  --source-run-slug image_dl_finetune_resnet18_specimen_select \
  --run-slug image_dl_resnet18_tta_bias_specimen_select \
  --model resnet18 \
  --split-mode specimen \
  --val-size 0.2 \
  --image-size 224 \
  --batch-size 64 \
  --bias-penalty 0.02
```

### Dependencies

`torch`, `torchvision`, `PIL`, `numpy`, `pandas`, `scikit-learn`, `tqdm`

### Mapping code

| Bước | Hàm | File |
|------|-----|------|
| Root / manifest / split | `resolve_root`, `load_image_manifest`, `split_train_val` | `train_image_handcrafted_ml_select_final_test.py` |
| Bias | `tune_class_bias`, `predict_with_bias` | same |
| Report | `make_report_row` | same |
| Dataset / transforms / model / train | `ImageDataset`, `train_transform`, `make_model`, `main` | `train_image_dl_finetune_select_final_test.py` |
| TTA + select | `make_transform`, `recipe_variants`, `predict_proba`, `main` | `tune_image_dl_checkpoint_bias_tta_select_final_test.py` |

---

## 10. Inference 1 ảnh

Config đã lock (best run):

```python
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn

ID2LABEL = {0: "ambient", 1: "leaf", 2: "trunk", 3: "twig"}
BIAS = torch.tensor([0.0, 0.0, -1.2, -1.0])  # locked

eval_tf = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def build_model(ckpt_path: str, device: str = "cuda"):
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 4))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model

@torch.inference_mode()
def predict_image(model, path: str, device: str = "cuda") -> dict:
    image = Image.open(path).convert("RGB")
    x = eval_tf(image).unsqueeze(0).to(device)
    logits = model(x)
    proba = torch.softmax(logits.float(), dim=1)[0].cpu()
    scores = torch.log(proba.clamp_min(1e-12)) + BIAS
    pred_id = int(scores.argmax().item())
    return {
        "pred_id": pred_id,
        "pred_label": ID2LABEL[pred_id],
        "proba": {ID2LABEL[i]: float(proba[i]) for i in range(4)},
    }
```

Checkpoint path:

```
outputs/audio_feature_benchmarks/image_dl_finetune_resnet18_specimen_select/models/
  image_dl_finetune_resnet18_specimen_select_best_val_model.pt
```

---

## 11. Metrics

**Primary (selection + claim):** 4-class Macro F1  

\[
\text{MacroF1} = \frac{1}{4}\sum_{c=0}^{3} F1_c
\]

**Secondary trong report:** accuracy 4-class, contact Macro F1 (leaf/trunk/twig), binary ambient-vs-contact Macro F1, per-class P/R/F1/support.

| Phase | Data | Mục đích |
|-------|------|----------|
| Selection | hand val | epoch, recipe, bias |
| Claim | robot **1 lần** sau lock | số báo cáo |

---

## 12. Giới hạn

1. **Exact image multi-label:** cùng pixels ↔ nhiều label (contact state từ audio).  
2. **Oracle ceiling robot** ~0.65 (majority per exact-hash) — best model 0.45.  
3. **Domain shift** hand camera → robot camera: val 0.58 thường rơi xuống ~0.45.  
4. **Bias lớn** dễ overfit hand prior → dùng `regularized_val_score`.  
5. Target clean **> 0.5** chưa đạt với image-only; cần audio/multimodal để vượt.

Chi tiết experiment history: `IMAGE_ONLY_DL_ATTEMPT_LOG.md`.

---

## Diagram tóm tắt (best path)

```
hand images
    → ResNet18 ImageNet fine-tune (specimen split)
    → best val epoch checkpoint                    [LOCK #1]
    → hand val: TTA recipes × (none|tuned bias)
    → max regularized score → plain + bias         [LOCK #2]
    → robot images → softmax → log+bias → argmax
    → Macro F1 = 0.450474
```

---

*Best clean image-only robot Macro F1: **0.450474** (`image_dl_resnet18_tta_bias_specimen_select`).*
