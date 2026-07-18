# Báo cáo tổng hợp phiên làm việc — Diagram Audio Pipeline Branch Ablation

**Ngày:** 2026-07-18  
**Workspace:** `tree_audio`  
**Phạm vi:** Toàn bộ phiên chat từ lúc xem diagram pipeline → multimodal ResNet18 + 6-view audio → các nhánh Weighted TTA / Anchor Blend / Report Gate / Final Source Blend / Segment Aggregation / Segment Consensus / Contact mass adjustment.

---

## 1. Tóm tắt điều hành

Phiên này bắt đầu từ diagram pipeline audio (diagram “Train/Val split + Extract feature”), xác nhận đây là pipeline **audio-only** (không image trong diagram gốc). Sau đó triển khai một loạt thí nghiệm **feature-level classical ML** theo từng node diagram, với protocol thống nhất:

1. Feature = ma trận total240 theo view (240-D mỗi view), fuse ngang.
2. **Group-safe** train/val trên hand only.
3. Hai variant: **full features** + **feature selection** (score trên train, chọn k trên val).
4. **Selection lock** với `test_loaded: false` **trước** khi load robot/test.
5. Robot/test đánh giá **đúng 1 lần** / variant; in đầy đủ macro F1, contact F1, binary F1, accuracy, confusion, per-class.

### Kết quả robot/test nổi bật

| Hạng | Thí nghiệm | Dim | Variant tốt nhất | **macro F1** | Acc |
|---:|---|---:|---|---:|---:|
| 1 | Multimodal 6-view audio + ResNet18-224 | 1952 | full ExtraTrees | **0.7053** | 0.7972 |
| 2 | Họ 6-view 1440-D (Anchor/ReportGate/FSB/SegAgg/SegCons/ContactMass) selected | 1440 (k=1200) | HistGB selected | **0.5633** | 0.7116 |
| 3 | Audio 6-view full / sixview slug | 1440 | HistGB full | **0.5583** | 0.7102 |
| 4 | Weighted-TTA high-SR only 3-view | 720 | selected k=600 | **0.5457** | 0.7035 |

> **Lưu ý quan trọng:** Các run 1440-D (Anchor → Contact mass) dùng **cùng ma trận feature 6×240** và cùng seed; điểm số gần như trùng nhau. Chúng được giữ **riêng theo protocol/diagram node**, không phải feature engineering độc lập mới mỗi lần.

Multimodal `selected_mm` (k=720, MLP) **tụt mạnh** trên test (0.5066) dù val cao — báo cáo trung thực cả full lẫn selected.

---

## 2. Protocol chống leakage (áp dụng mọi thí nghiệm feature-level)

| Quy tắc | Cách thực thi |
|---|---|
| Không dùng test khi chọn model / feature | API `select_feature_mask_val_only` / `train_and_select_model_on_val` raise `TestDataLeakageError` nếu truyền `X_test`/`y_test` |
| Train/val group-disjoint | `StratifiedGroupKFold` + `assert_group_safe_split` |
| Lock trước test | File `locks/selected_without_test.json` ghi `test_loaded: false`, mtime ≤ final reports |
| Test 1 lần | `evaluation_count: 1` mỗi variant |
| Không filename class-token features | Lock ghi `filename_class_tokens_used: false` |

**Phân biệt với paper audio-only 0.7027:** bản paper-safe best là **probability blend** (anchor lift + report_gate_onehot + segment lift). Phiên này **không** thay thế stack đó; đây là **feature-level ML** trên total240 views.

---

## 3. Kiến trúc code dùng chung

### Core pure helpers — `audio_six_view_fuse_core.py`

- View constants: `SIX_VIEWS` (6×240 → 1440), `HIGHSR_TTA_VIEWS` (3×240 → 720), `ANCHOR_BLEND_*`, `REPORT_GATE_*`, `FINAL_SOURCE_BLEND_*`, `SEGMENT_AGGREGATION_*`, `SEGMENT_CONSENSUS_*`, `CONTACT_MASS_*`
- Multimodal: `IMAGE_DIM=512`, `MULTIMODAL_DIM=1952`
- Fuse: `fuse_view_features`, `fuse_highsr_tta_views`, `fuse_anchor_blend_parent_views`, …
- Selection: `select_feature_mask_val_only` (train scores + val k), `train_and_select_model_on_val`
- Model zoo: HistGradientBoosting, ExtraTrees, LogisticRegression (+ optional MLP cho multimodal)
- Metrics: `metrics_from_pred` / full metrics trong train scripts

### Pattern orchestration (mọi train script)

```
load hand features → group-safe split → full ML grid on val
→ feature mask (train scores, val k) → ML grid on selected
→ refit on full hand → write lock (test_loaded=false)
→ load robot → evaluate once per variant → reports/CSV/confusion
```

---

## 4. Diễn biến phiên theo thời gian

### 4.1. Xác nhận diagram

- User hỏi diagram có phải “only audio” không.
- **Kết luận:** đúng audio-only paper-safe style: 16 kHz + 44.1 kHz high-SR, stress views clean/robot_mix/bandlimit, feature MFCC/STFT/FFT/Mel, pairwise + multi-class OOF, Weighted TTA, Anchor Blend, Report Gate, Final Source Blend, segment postprocess. **Không** có nhánh image trong diagram.

### 4.2. Multimodal: 6-view audio (1440) + ResNet18 (512)

- Goal: fuse 6×240 + ResNet18-224 frozen embeddings → ML/MLP, full + selection.
- Entry: `train_sixview_resnet18_mm_fuse_select_final_test.py`
- Tests: `tests/test_sixview_resnet18_mm_fuse.py`

#### Kết quả robot/test — Multimodal

| Metric | **full_mm (1952-D)** | **selected_mm** |
|---|---:|---:|
| macro F1 4-class | **0.7053** | **0.5066** |
| weighted F1 | — | — |
| macro precision | — | — |
| macro recall | — | — |
| contact macro F1 | 0.6280 | 0.3723 |
| binary macro F1 | 0.9310 | 0.8971 |
| accuracy | 0.7972 | 0.6985 |
| n_test | 2219 | 2219 |
| evaluation_count | 1 | 1 |
| model (locked) | ExtraTrees | MLP |
| val macro F1 | 0.8451 | 0.8777 |
| feature dim / k | 1952 | 720 (k=720) |

**Per-class full_mm:**

_Không có per_class trong report JSON (chỉ macro metrics)._

**Confusion full_mm:**

```
          amb  leaf trunk twig
ambient   1132     0     0     0
leaf        20   189     4    80
trunk      110    80   201    70
twig        22    64     0   247
```

### 4.3. Audio-only 6-view full (240×6 → 1440)

- Goal: chỉ audio 6 view, full + selection.
- Entry: `train_audio_sixview_fuse_select_final_test.py`
- Cache path: sr16 clean/stress + high-SR total240@44100 (default `--from-caches`)

| Metric | **full_240d_x6** | **selected_240d_x6 (k=1440 all)** |
|---|---:|---:|
| macro F1 4-class | **0.5583** | **0.5583** |
| weighted F1 | — | — |
| macro precision | — | — |
| macro recall | — | — |
| contact macro F1 | 0.4276 | 0.4276 |
| binary macro F1 | 0.9465 | 0.9465 |
| accuracy | 0.7102 | 0.7102 |
| n_test | 2219 | 2219 |
| evaluation_count | 1 | 1 |
| model (locked) | HistGradientBoosting | HistGradientBoosting |
| val macro F1 | 0.8203 | 0.8203 |
| feature dim / k | 1440 | 1440 (k=1440) |

> selected chọn k=1440 (full) → hai variant trùng điểm trên test.

### 4.4. Weighted TTA blend (high-SR only, 240×3 → 720)

- Diagram: nhánh xanh 44.1 kHz → Clean/Robo-mix/Bandlimit → Weighted TTA.
- **Chỉ 3 view high-SR**, không sr16.
- Entry: `train_audio_wtt_highsr3_fuse_select_final_test.py`

| Metric | **full_240d_x3 (720-D)** | **selected k=600** |
|---|---:|---:|
| macro F1 4-class | **0.5434** | **0.5457** |
| weighted F1 | 0.6886 | 0.6878 |
| macro precision | 0.5603 | 0.5637 |
| macro recall | 0.5443 | 0.5478 |
| contact macro F1 | 0.4083 | 0.4114 |
| binary macro F1 | 0.9443 | 0.9443 |
| accuracy | 0.7044 | 0.7035 |
| n_test | 2219 | 2219 |
| evaluation_count | 1 | 1 |
| model (locked) | HistGradientBoosting | HistGradientBoosting |
| val macro F1 | 0.8001 | 0.8009 |
| feature dim / k | 720 | 600 (k=600) |

**Per-class full:**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ambient | 0.9020 | 1.0000 | 0.9485 | 1132 |
| leaf | 0.3828 | 0.2730 | 0.3187 | 293 |
| trunk | 0.5643 | 0.3905 | 0.4615 | 461 |
| twig | 0.3922 | 0.5135 | 0.4447 | 333 |

**Confusion full:**

```
          amb  leaf trunk twig
ambient   1132     0     0     0
leaf        14    80    70   129
trunk       92    53   180   136
twig        17    76    69   171
```

---

## 5. Các nhánh diagram 1440-D (feature-level parents)

Các mục dưới đây dùng **cùng 6 view total240**, dim **1440**, model locked **HistGradientBoosting**, selected **k=1200**. Điểm test gần như giống nhau; khác nhau chủ yếu ở **protocol naming** (map đúng node diagram) và `excludes_downstream` / `includes_stage`.

### 5.1. Anchor Blend

- **Diagram / scope:** Pairwise (sr16×3) + Weighted TTA (highsr×3)
- **Entry:** `train_audio_anchor_blend_fuse_select_final_test.py`
- **Tests:** `tests/test_audio_anchor_blend_fuse.py`
- **Output slug:** `outputs/audio_feature_benchmarks/audio_anchor_blend_fuse_select/`
- **Parents (lock):** `{"pairwise_sr16": ["sr16_clean", "sr16_robot_mix", "sr16_bandlimit"], "weighted_tta_highsr": ["highsr_clean", "highsr_robot_mix", "highsr_bandlimit"]}`

| Metric | **full_240d_x6** | **selected k=1200** |
|---|---:|---:|
| macro F1 4-class | **0.5583** | **0.5633** |
| weighted F1 | 0.6967 | 0.6979 |
| macro precision | 0.5777 | 0.5780 |
| macro recall | 0.5573 | 0.5623 |
| contact macro F1 | 0.4276 | 0.4347 |
| binary macro F1 | 0.9465 | 0.9452 |
| accuracy | 0.7102 | 0.7116 |
| n_test | 2219 | 2219 |
| evaluation_count | 1 | 1 |
| model (locked) | HistGradientBoosting | HistGradientBoosting |
| val macro F1 | 0.8203 | 0.8229 |
| feature dim / k | 1440 | 1200 (k=1200) |

**Per-class (full):**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ambient | 0.9056 | 1.0000 | 0.9505 | 1132 |
| leaf | 0.4474 | 0.3481 | 0.3916 | 293 |
| trunk | 0.5757 | 0.3796 | 0.4575 | 461 |
| twig | 0.3822 | 0.5015 | 0.4338 | 333 |

**Confusion (full) — đại diện cho cả họ 1440-D:**

```
          amb  leaf trunk twig
ambient   1132     0     0     0
leaf        13   102    56   122
trunk       88    50   175   148
twig        17    76    73   167
```

### 5.2. Report Gate / Ensemble

- **Diagram / scope:** Multi-class OOF (sr16×3) + Weighted TTA (highsr×3)
- **Entry:** `train_audio_report_gate_fuse_select_final_test.py`
- **Tests:** `tests/test_audio_report_gate_fuse.py`
- **Output slug:** `outputs/audio_feature_benchmarks/audio_report_gate_fuse_select/`
- **Parents (lock):** `{"multiclass_sr16": ["sr16_clean", "sr16_robot_mix", "sr16_bandlimit"], "weighted_tta_highsr": ["highsr_clean", "highsr_robot_mix", "highsr_bandlimit"]}`

| Metric | **full_240d_x6** | **selected k=1200** |
|---|---:|---:|
| macro F1 4-class | **0.5583** | **0.5633** |
| weighted F1 | 0.6967 | 0.6979 |
| macro precision | 0.5777 | 0.5780 |
| macro recall | 0.5573 | 0.5623 |
| contact macro F1 | 0.4276 | 0.4347 |
| binary macro F1 | 0.9465 | 0.9452 |
| accuracy | 0.7102 | 0.7116 |
| n_test | 2219 | 2219 |
| evaluation_count | 1 | 1 |
| model (locked) | HistGradientBoosting | HistGradientBoosting |
| val macro F1 | 0.8203 | 0.8229 |
| feature dim / k | 1440 | 1200 (k=1200) |

**Per-class (full):**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ambient | 0.9056 | 1.0000 | 0.9505 | 1132 |
| leaf | 0.4474 | 0.3481 | 0.3916 | 293 |
| trunk | 0.5757 | 0.3796 | 0.4575 | 461 |
| twig | 0.3822 | 0.5015 | 0.4338 | 333 |

### 5.3. Final Source Blend

- **Diagram / scope:** Union parents Anchor + Report Gate (sr16×3 + highsr×3)
- **Entry:** `train_audio_final_source_blend_fuse_select_final_test.py`
- **Tests:** `tests/test_audio_final_source_blend_fuse.py`
- **Output slug:** `outputs/audio_feature_benchmarks/audio_final_source_blend_fuse_select/`
- **Parents (lock):** `{"anchor_side_sr16": ["sr16_clean", "sr16_robot_mix", "sr16_bandlimit"], "report_gate_side_sr16": ["sr16_clean", "sr16_robot_mix", "sr16_bandlimit"], "weighted_tta_highsr": ["highsr_clean", "highsr_robot_mix", "highsr_bandlimit"]}`

| Metric | **full_240d_x6** | **selected k=1200** |
|---|---:|---:|
| macro F1 4-class | **0.5583** | **0.5633** |
| weighted F1 | 0.6967 | 0.6979 |
| macro precision | 0.5777 | 0.5780 |
| macro recall | 0.5573 | 0.5623 |
| contact macro F1 | 0.4276 | 0.4347 |
| binary macro F1 | 0.9465 | 0.9452 |
| accuracy | 0.7102 | 0.7116 |
| n_test | 2219 | 2219 |
| evaluation_count | 1 | 1 |
| model (locked) | HistGradientBoosting | HistGradientBoosting |
| val macro F1 | 0.8203 | 0.8229 |
| feature dim / k | 1440 | 1200 (k=1200) |

**Per-class (full):**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ambient | 0.9056 | 1.0000 | 0.9505 | 1132 |
| leaf | 0.4474 | 0.3481 | 0.3916 | 293 |
| trunk | 0.5757 | 0.3796 | 0.4575 | 461 |
| twig | 0.3822 | 0.5015 | 0.4338 | 333 |

### 5.4. Segment Aggregation

- **Diagram / scope:** Upstream FSB six views; **không** Consensus/Contact/FinalClass
- **Entry:** `train_audio_segment_aggregation_fuse_select_final_test.py`
- **Tests:** `tests/test_audio_segment_aggregation_fuse.py`
- **Output slug:** `outputs/audio_feature_benchmarks/audio_segment_aggregation_fuse_select/`
- **Parents (lock):** `{"upstream_sr16": ["sr16_clean", "sr16_robot_mix", "sr16_bandlimit"], "upstream_highsr": ["highsr_clean", "highsr_robot_mix", "highsr_bandlimit"], "fed_by": "final_source_blend", "excludes_downstream": ["segment_consensus", "contact_mass_adjustment", "final_class_decision"]}`

| Metric | **full_240d_x6** | **selected k=1200** |
|---|---:|---:|
| macro F1 4-class | **0.5583** | **0.5633** |
| weighted F1 | 0.6967 | 0.6979 |
| macro precision | 0.5777 | 0.5780 |
| macro recall | 0.5573 | 0.5623 |
| contact macro F1 | 0.4276 | 0.4347 |
| binary macro F1 | 0.9465 | 0.9452 |
| accuracy | 0.7102 | 0.7116 |
| n_test | 2219 | 2219 |
| evaluation_count | 1 | 1 |
| model (locked) | HistGradientBoosting | HistGradientBoosting |
| val macro F1 | 0.8203 | 0.8229 |
| feature dim / k | 1440 | 1200 (k=1200) |

**Per-class (full):**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ambient | 0.9056 | 1.0000 | 0.9505 | 1132 |
| leaf | 0.4474 | 0.3481 | 0.3916 | 293 |
| trunk | 0.5757 | 0.3796 | 0.4575 | 461 |
| twig | 0.3822 | 0.5015 | 0.4338 | 333 |

### 5.5. Segment Consensus

- **Diagram / scope:** Upstream SegAgg; includes consensus; **không** contact-mass/final-class
- **Entry:** `train_audio_segment_consensus_fuse_select_final_test.py`
- **Tests:** `tests/test_audio_segment_consensus_fuse.py`
- **Output slug:** `outputs/audio_feature_benchmarks/audio_segment_consensus_fuse_select/`
- **Parents (lock):** `{"upstream_sr16": ["sr16_clean", "sr16_robot_mix", "sr16_bandlimit"], "upstream_highsr": ["highsr_clean", "highsr_robot_mix", "highsr_bandlimit"], "fed_by": "segment_aggregation", "includes_stage": "segment_consensus", "excludes_downstream": ["contact_mass_adjustment", "final_class_decision"]}`

| Metric | **full_240d_x6** | **selected k=1200** |
|---|---:|---:|
| macro F1 4-class | **0.5583** | **0.5633** |
| weighted F1 | 0.6967 | 0.6979 |
| macro precision | 0.5777 | 0.5780 |
| macro recall | 0.5573 | 0.5623 |
| contact macro F1 | 0.4276 | 0.4347 |
| binary macro F1 | 0.9465 | 0.9452 |
| accuracy | 0.7102 | 0.7116 |
| n_test | 2219 | 2219 |
| evaluation_count | 1 | 1 |
| model (locked) | HistGradientBoosting | HistGradientBoosting |
| val macro F1 | 0.8203 | 0.8229 |
| feature dim / k | 1440 | 1200 (k=1200) |

**Per-class (full):**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ambient | 0.9056 | 1.0000 | 0.9505 | 1132 |
| leaf | 0.4474 | 0.3481 | 0.3916 | 293 |
| trunk | 0.5757 | 0.3796 | 0.4575 | 461 |
| twig | 0.3822 | 0.5015 | 0.4338 | 333 |

### 5.6. Contact mass adjustment

- **Diagram / scope:** Upstream SegCons; includes contact-mass; **không** Final Class Decision
- **Entry:** `train_audio_contact_mass_fuse_select_final_test.py`
- **Tests:** `tests/test_audio_contact_mass_fuse.py`
- **Output slug:** `outputs/audio_feature_benchmarks/audio_contact_mass_fuse_select/`
- **Parents (lock):** `{"upstream_sr16": ["sr16_clean", "sr16_robot_mix", "sr16_bandlimit"], "upstream_highsr": ["highsr_clean", "highsr_robot_mix", "highsr_bandlimit"], "fed_by": "segment_consensus", "includes_stage": "contact_mass_adjustment", "excludes_downstream": ["final_class_decision"]}`

| Metric | **full_240d_x6** | **selected k=1200** |
|---|---:|---:|
| macro F1 4-class | **0.5583** | **0.5633** |
| weighted F1 | 0.6967 | 0.6979 |
| macro precision | 0.5777 | 0.5780 |
| macro recall | 0.5573 | 0.5623 |
| contact macro F1 | 0.4276 | 0.4347 |
| binary macro F1 | 0.9465 | 0.9452 |
| accuracy | 0.7102 | 0.7116 |
| n_test | 2219 | 2219 |
| evaluation_count | 1 | 1 |
| model (locked) | HistGradientBoosting | HistGradientBoosting |
| val macro F1 | 0.8203 | 0.8229 |
| feature dim / k | 1440 | 1200 (k=1200) |

**Per-class (full):**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ambient | 0.9056 | 1.0000 | 0.9505 | 1132 |
| leaf | 0.4474 | 0.3481 | 0.3916 | 293 |
| trunk | 0.5757 | 0.3796 | 0.4575 | 461 |
| twig | 0.3822 | 0.5015 | 0.4338 | 333 |


---

## 6. Bảng so sánh tổng hợp robot/test

| # | Diagram node | Feature scope | Dim | Full macro F1 | Selected macro F1 | Full Acc | Selected Acc |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Multimodal + ResNet18 | 6×240 audio + 512 image | 1952 | 0.7053 | 0.5066 | 0.7972 | 0.6985 |
| 2 | Audio 6-view (sixview slug) | sr16×3 + highsr×3 | 1440 | 0.5583 | 0.5583 | 0.7102 | 0.7102 |
| 3 | Weighted TTA (high-SR only) | highsr×3 | 720 | 0.5434 | 0.5457 | 0.7044 | 0.7035 |
| 4 | Anchor Blend parents | sr16×3 + highsr×3 | 1440 | 0.5583 | 0.5633 | 0.7102 | 0.7116 |
| 5 | Report Gate parents | sr16×3 + highsr×3 | 1440 | 0.5583 | 0.5633 | 0.7102 | 0.7116 |
| 6 | Final Source Blend parents | sr16×3 + highsr×3 | 1440 | 0.5583 | 0.5633 | 0.7102 | 0.7116 |
| 7 | Segment Aggregation only | sr16×3 + highsr×3 | 1440 | 0.5583 | 0.5633 | 0.7102 | 0.7116 |
| 8 | Segment Consensus only | sr16×3 + highsr×3 | 1440 | 0.5583 | 0.5633 | 0.7102 | 0.7116 |
| 9 | Contact mass adjustment only | sr16×3 + highsr×3 | 1440 | 0.5583 | 0.5633 | 0.7102 | 0.7116 |

### Đọc bảng

1. **Multimodal full (0.705)** vượt rõ audio-only feature ML (~0.56) — image ResNet18 đóng góp lớn ở feature-level fusion (khác stack proba 0.7027).
2. **High-SR 3-view only (~0.546)** yếu hơn 6-view (~0.558–0.563) → sr16 stress views có thông tin bổ sung.
3. **Feature selection** trên họ 1440-D (k=1200) nhích nhẹ full (0.563 vs 0.558); trên multimodal thì selection (MLP, k=720) **hỏng** generalization.
4. Các node diagram sau Final Source Blend (SegAgg/Cons/ContactMass) ở **feature-level parent-view** không tạo representation mới — điểm số trùng họ 1440-D là đúng kỳ vọng.

---

## 7. Index artifacts

### 7.1. Scripts / core

| File | Vai trò |
|---|---|
| `audio_six_view_fuse_core.py` | Pure fuse/select/leakage/model zoo |
| `train_audio_sixview_fuse_select_final_test.py` | Audio 6-view 1440 |
| `train_sixview_resnet18_mm_fuse_select_final_test.py` | Multimodal 1952 |
| `train_audio_wtt_highsr3_fuse_select_final_test.py` | Weighted TTA high-SR 720 |
| `train_audio_anchor_blend_fuse_select_final_test.py` | Anchor Blend parents |
| `train_audio_report_gate_fuse_select_final_test.py` | Report Gate parents |
| `train_audio_final_source_blend_fuse_select_final_test.py` | Final Source Blend parents |
| `train_audio_segment_aggregation_fuse_select_final_test.py` | Segment Aggregation only |
| `train_audio_segment_consensus_fuse_select_final_test.py` | Segment Consensus only |
| `train_audio_contact_mass_fuse_select_final_test.py` | Contact mass only |

### 7.2. Tests

| Test module |
|---|
| `tests/test_audio_sixview_fuse.py` |
| `tests/test_sixview_resnet18_mm_fuse.py` |
| `tests/test_audio_wtt_highsr3_fuse.py` |
| `tests/test_audio_anchor_blend_fuse.py` |
| `tests/test_audio_report_gate_fuse.py` |
| `tests/test_audio_final_source_blend_fuse.py` |
| `tests/test_audio_segment_aggregation_fuse.py` |
| `tests/test_audio_segment_consensus_fuse.py` |
| `tests/test_audio_contact_mass_fuse.py` |

### 7.3. Output run directories

| Run slug | under `outputs/audio_feature_benchmarks/` |
|---|---|
| `sixview_resnet18_mm_fuse_select` | Multimodal |
| `audio_sixview_fuse_select` | Audio 6-view |
| `audio_wtt_highsr3_fuse_select` | WTT high-SR3 |
| `audio_anchor_blend_fuse_select` | Anchor |
| `audio_report_gate_fuse_select` | Report Gate |
| `audio_final_source_blend_fuse_select` | Final Source Blend |
| `audio_segment_aggregation_fuse_select` | Segment Aggregation |
| `audio_segment_consensus_fuse_select` | Segment Consensus |
| `audio_contact_mass_fuse_select` | Contact mass |

Mỗi run có: `locks/selected_without_test.json`, `reports/*_final_test_report.json`, predictions CSV, confusion CSV, `method_card_before_test.json`, `run_summary.json`.

### 7.4. Feature caches dùng lại

| Cache | Vai trò |
|---|---|
| `outputs/audio_feature_benchmarks/total240_trainval_select/features/{hand_train_full,robot_test}` | sr16 clean 240-D |
| `.../total240_stress_cv_select/stress_features/{robot_mix,bandlimit}` | sr16 stress hand |
| `.../audio_tta_contact_stress_cv_select/test_tta_features/{robot_mix,bandlimit}` | sr16 stress robot |
| `outputs/audio_sample_rate_ablation/all44100/.../total240_sr44100_caches/` | high-SR clean + stress 240-D |
| `outputs/image_deep_features/resnet18_224/{hand_train_full,robot_test}` | ResNet18 512-D |

---

## 8. Sơ đồ map diagram → thí nghiệm

```
Raw Audio
  ├─ Mono+16k ── Clean / Robo-mix / Bandlimit ──┐
  │                                              ├─ 6×240 fuse ──► [Audio 6-view / Anchor / ReportGate /
  └─ 44.1k high-SR ── Clean / Robo-mix / Bandlimit ┘                 FSB / SegAgg / SegCons / ContactMass]
         │
         └─ (chỉ 3 high-SR) ──► [Weighted TTA high-SR 720-D experiment]

  6×240 audio + ResNet18 512 ──► [Multimodal 1952-D]  ★ best feature-level F1 0.705

Diagram proba stack (Anchor Blend → Report Gate → Final Source Blend → Segment* → Contact mass)
  → phiên này: feature-level proxy theo parent views, KHÔNG re-run proba 0.7027 lift stack
```

---

## 9. Kết luận & khuyến nghị

1. **Feature-level multimodal** (6-view audio + frozen ResNet18) đạt **macro F1 ≈ 0.705** — ngang/hơi hơn paper audio-only proba stack 0.7027, nhưng là pipeline **khác** (feature ML vs proba blend).
2. **Audio feature-only ML** trên total240 6-view dừng quanh **0.56–0.56** macro F1; high-SR only ~0.55.
3. **Feature selection univariate** hữu ích nhẹ trên 1440-D (k=1200) nhưng **nguy hiểm** khi kết hợp MLP trên multimodal (val cao, test sập).
4. Các node diagram sau TTA (Anchor…Contact mass) nên hiểu trong phiên này là **ablation theo protocol/node**, không phải representation mới — để so sánh proba stack thật, cần pipeline OOF blend (ngoài scope feature-level).
5. Protocol **lock-before-test** đã được unit-test + entry-path gate trên mọi nhánh.

---

## 10. Metadata báo cáo

- Nguồn số: JSON trong `outputs/audio_feature_benchmarks/*/reports/` và `locks/selected_without_test.json`
- Snapshot trung gian: `/tmp/grok-goal-438ca6e042e5/implementer/session_runs_snapshot.json`
- Không re-train trong bước viết báo cáo; chỉ đọc artifact có sẵn.
