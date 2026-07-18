# Audio Log-Consensus Pair Blend Select — Final Test

**Protocol**: Audio-only segment-consistency blend with log-probability consensus.
**Entry point**: `train_audio_log_consensus_pair_blend_select_final_test.py`
**Date**: 2026-07-06

---

## 1. Overview

This experiment blends two pre-trained audio models — High-SR Temporal TTA and Pairwise Contact Stress CV — using out-of-fold (OOF) predictions from the **hand/default train set only**. A single hyperparameter (blend weight) and a single group-decoding rule (`sum_log_proba`) are selected without any access to robot/test data.

After selection is **locked**, the frozen configuration is applied to the robot/test set exactly once.

### Key design constraints (anti-leakage):

| Rule | Enforced by |
|---|---|
| No image / vision features | `total240` feature set = MFCC + STFT + Mel + FFT (audio only) |
| No test labels before lock | `write_json(selection_lock)` before `load_final_frame_and_proba()` |
| No test predictions in selection | Only `y_train` + OOF (cross-val on train) used in `evaluate_candidates()` |
| Models treated as frozen artifacts | High-SR & pairwise predictions loaded from prior experiments (no retrain) |

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│          train_audio_log_consensus_pair_blend_select         │
│                    _final_test.py                            │
│                                                              │
│  Override: HIGHSR_WEIGHT_CANDIDATES = [0.65, 0.70, 0.75, 0.80]│
│  Override: GROUP_RULES = ["sum_log_proba"]                  │
│  Override: --run-slug = "audio_log_consensus_pair_blend_select"│
│                                                              │
│  └─▸ segment_impl.main()       (398-line implementation)     │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: SETUP                            │
├─────────────────────────────────────────────────────────────┤
│  base.configure_feature_set("total240")                      │
│    → 240-dim audio features (MFCC+STFT+Mel+FFT × 2 crops)   │
│  Create run/report/model directories                         │
│  Load hand/default dataset.csv → train_df (10676 samples)    │
│  Load audio features from cache: (10676, 240)                │
│  Load 4-fold CV split assignments from pairwise experiment   │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                PHASE 2: LOAD OOF PREDICTIONS                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─── load_highsr_oof() ───────────────────────────────────┐│
│  │  Read lock JSON → selected_candidate + TTA weights      ││
│  │  Load OOF proba per audio view (lr, hr, log, mel, etc.) ││
│  │  Weighted blend of views → normalize                    ││
│  │  Output: (10676, 4) proba matrix (train OOF)            ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─── load_or_rebuild_pairwise_oof() ──────────────────────┐│
│  │  Load clean audio features + stress audio features      ││
│  │  Read pairwise lock → selected model candidate          ││
│  │  For each fold (4-fold CV):                             ││
│  │    fit_pairwise_candidate(train_idx)                     ││
│  │    predict(val_idx) → accumulate OOF                    ││
│  │  Cache result to {run_dir}/oof_sources/*.npy            ││
│  │  Output: (10676, 4) proba matrix (train OOF)            ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│        PHASE 3: OOF SELECTION (TRAIN ONLY — NO TEST)        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  evaluate_candidates(highsr_oof, pairwise_oof, y_train):     │
│                                                              │
│  For each highsr_weight ∈ {0.65, 0.70, 0.75, 0.80}:         │
│    pairwise_weight = 1 - highsr_weight                       │
│    blended = normalize(weight_a × oof_a + weight_b × oof_b) │
│                                                              │
│    group_decode(blended, "sum_log_proba"):                   │
│      group_codes = factorize(frame["group_key"])             │
│      log_proba = np.log(clip(blended, 1e-12, 1.0))          │
│      sum_log = bincount(group_codes, weight=log_proba)       │
│      pred = argmax over groups                               │
│                                                              │
│    Score:                                                     │
│      macro_f1     (4-class: ambient, leaf, trunk, twig)     │
│      contact_macro_f1 (3-class: leaf, trunk, twig)          │
│      binary_macro_f1  (normal vs contact)                    │
│      selection_score = 0.65 × macro + 0.25 × contact        │
│                       + 0.10 × binary                        │
│                                                              │
│  Sort by selection_score descending → leaderboard            │
│                                                              │
│  ████████████ SELECTION LOCK WRITTEN TO DISK █████████████   │
│  ██████████ NO TEST DATA TOUCHED ABOVE THIS LINE ██████████   │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│        PHASE 4: TEST PREDICTION (FROZEN CONFIG)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Load highsr final test predictions CSV                      │
│    → highsr_final_proba: (2219, 4)                           │
│  Load pairwise final test predictions CSV                    │
│    → pair_final_proba: (2219, 4)                             │
│  Verify alignment on audio_file & y                          │
│                                                              │
│  final_blend = normalize(0.75 × highsr_final                 │
│                        + 0.25 × pair_final)                  │
│                                                              │
│  final_pred = group_decode(frame, final_blend,               │
│                            "sum_log_proba")                   │
│                                                              │
│  Score: y_test vs final_pred                                 │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                PHASE 5: ARTIFACTS                             │
├─────────────────────────────────────────────────────────────┤
│  final_test_report.csv        (1-row summary)                │
│  final_test_predictions.csv   (per-sample predictions)       │
│  confusion_matrix.csv          (4×4 matrix)                  │
│  selected_model_bundle.joblib  (full metadata bundle)        │
│  protocol_summary.json         (complete audit trail)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Code Dependency Graph

```
train_audio_log_consensus_pair_blend_select_final_test.py   ← ENTRY (config wrapper)
│
└── train_audio_group_consistency_pair_blend_select_final_test.py   ← MAIN (398 lines)
    │
    ├── train_audio_highsr_temporal_tta_select_final_test.py        ← High-SR model defs
    │   ├── train_audio_multifeature_tta_grid_ensemble_select_final_test.py
    │   │   └── train_audio_tta_grid_hgb_select_final_test.py
    │   │       └── train_audio_tta_contact_stress_cv_select_final_test.py
    │   ├── train_cv_select_final_test.py
    │   ├── train_stress_cv_select_final_test.py
    │   └── train_val_select_final_test.py
    │
    ├── train_audio_pairwise_contact_stress_cv_select_final_test.py ← Pairwise model defs
    │   ├── train_cv_select_final_test.py
    │   ├── train_stress_cv_select_final_test.py
    │   └── train_val_select_final_test.py
    │
    ├── train_cv_select_final_test.py                              ← CV + ensemble framework
    │   └── train_val_select_final_test.py
    │
    ├── train_stress_cv_select_final_test.py                       ← Stress audio features
    │   ├── train_cv_select_final_test.py
    │   └── train_val_select_final_test.py
    │
    └── train_val_select_final_test.py                             ← BASE: config, features, utils
```

---

## 4. Feature Specification

| Feature Set | Dim | Composition |
|---|---|---|
| `total240` | 240 | `total120(full_signal)` + `total120(top_energy_window)` |
| `total120` | 120 | MFCC(40) + STFT(28) + Mel(28) + FFT(24) |
| `mfcc40` | 40 | MFCC mean + std (20 coeffs each) |
| `stft28` | 28 | RMS, ZCR, centroid, bandwidth, rolloff, flatness, flux × (mean, std, max, p90) |
| `mel28` | 28 | 7 Mel groups × (mean, std, max, p90) |
| `fft24` | 24 | 8 FFT bands + 8 log bands + 4 ratios + 4 extra (entropy, centroid, dominant freq, magnitude) |

All features extracted from audio `.wav` files via `librosa.load()` — **no images**.

---

## 5. Group Decoding: `sum_log_proba`

For each audio segment with multiple windows:

```
sum_log_proba per group = Σ log( clip(proba_window, 1e-12, 1.0) )
group_pred = argmax( sum_log_proba over 4 classes )
```

This treats each window as independent evidence and multiplies probabilities in log space (equivalent to product of probabilities), favoring consistent consensus across windows.

---

## 6. Results

### 6.1 OOF Leaderboard (Train-Only Selection)

| highsr_w | pair_w | group_rule | macro_f1 | contact_f1 | binary_f1 | selection_score |
|---|---|---|---|---|---|---|
| **0.75** | **0.25** | **sum_log_proba** | **0.876** | **0.837** | **0.991** | **0.878** |
| 0.80 | 0.20 | sum_log_proba | 0.874 | 0.835 | 0.991 | 0.876 |
| 0.70 | 0.30 | sum_log_proba | 0.874 | 0.835 | 0.991 | 0.876 |
| 0.65 | 0.35 | sum_log_proba | 0.871 | 0.831 | 0.991 | 0.873 |

**Selected**: highsr_weight=0.75, pairwise_weight=0.25, group_rule=sum_log_proba

### 6.2 Final Test Results (Robot/Test — 2219 samples)

| Metric | Value |
|---|---|
| Accuracy (4-class) | **0.765** |
| Macro F1 (4-class) | **0.654** |
| Contact Macro F1 (3-class) | **0.560** |
| Binary Macro F1 (normal vs contact) | **0.930** |
| Selected OOF Macro F1 | 0.876 |

**OOF → Test gap**: ~22 points on macro F1 (0.876 → 0.654)

### 6.3 Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| **ambient** (normal) | 0.881 | 1.000 | 0.937 | 1132 |
| leaf | 0.595 | 0.758 | 0.667 | 293 |
| trunk | 0.807 | 0.354 | 0.492 | 461 |
| twig | 0.501 | 0.541 | 0.520 | 333 |

### 6.4 Confusion Matrix

```
             ambient  leaf  trunk  twig
  ambient      1132     0      0     0
  leaf            2   222     15    54
  trunk         123    50    163   125
  twig           28   101     24   180
```

**Observation**: Ambient class perfect (1132/1132). Main confusion: trunk→ambient (123), twig→leaf (101), and trunk↔twig bidirectional confusion.

---

## 7. Output Artifacts

```
outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/
├── oof_sources/
│   └── pairwise_selected_clean_oof_proba.npy    ← cached pairwise OOF
├── reports/
│   ├── method_card_before_test.json             ← protocol & data metadata
│   ├── oof_group_leaderboard.csv                ← 4 candidates ranked
│   ├── selected_without_test.json               ← selection lock
│   ├── final_test_report.csv                    ← 1-row summary
│   ├── final_test_predictions.csv               ← per-sample predictions
│   ├── final_test_confusion_matrix.csv          ← 4×4 matrix
│   └── protocol_summary.json                    ← full audit trail
└── models/
    └── selected_model_bundle.joblib             ← metadata bundle
```

---

## 8. Key Functions

| Function | File | Purpose |
|---|---|---|
| `main()` | group_consistency | Orchestrator |
| `load_highsr_oof()` | group_consistency | Load High-SR OOF predictions (train only) |
| `load_or_rebuild_pairwise_oof()` | group_consistency | Load/rebuild pairwise OOF via 4-fold CV |
| `evaluate_candidates()` | group_consistency | Enumerate all blend weights × group rules, score on train OOF |
| `group_decode()` | group_consistency | Apply group-level consensus decoding (mean_proba or sum_log_proba) |
| `load_final_frame_and_proba()` | group_consistency | Load test predictions from prior experiments |
| `normalize()` | group_consistency | Clip + L1-normalize probability vectors |
| `fast_macro_f1()` | group_consistency | Vectorized macro F1 (any label subset) |
| `build_or_load_feature_cache()` | base | Cache audio features to disk with manifest signature |
| `extract_features_for_file()` | base | Extract `total240` features from a single audio file |
| `load_manifest()` | base | Load dataset.csv, validate, build audio paths |

---

## 9. Usage

```bash
cd /home/ttung05/Desktop/tree_audio

# Default run (uses cached OOF if available):
python train_audio_log_consensus_pair_blend_select_final_test.py

# Force rebuild pairwise OOF:
python train_audio_log_consensus_pair_blend_select_final_test.py \
  --force-rebuild-pairwise-oof

# Custom output directory:
python train_audio_log_consensus_pair_blend_select_final_test.py \
  --output /path/to/custom/outputs
```
