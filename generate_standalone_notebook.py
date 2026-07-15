#!/usr/bin/env python3
"""Generate a fully self-contained notebook from the audio_log_consensus experiment.

The notebook uses only pre-extracted features / cached OOF and does NOT import
any train_audio_*.py project files. All logic is inlined.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import nbformat as nbf

OUTPUT_PATH = Path(__file__).resolve().parent / "audio_log_consensus_pair_blend_standalone.ipynb"

# ── helpers ──────────────────────────────────────────────────────────
def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source)

def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)

# ── cells ────────────────────────────────────────────────────────────
cells: list[nbf.NotebookNode] = []

# ──────────────────────── CELL 0 : Title ─────────────────────────────
cells.append(md("""# Audio Log-Consensus Pair Blend Select — Final Test

## Self-contained standalone notebook

**Protocol:** Audio-only segment-consistency blend with log-probability consensus.

This notebook replicates **exactly** the logic of `train_audio_log_consensus_pair_blend_select_final_test.py`
but is **fully self-contained** — zero project imports. It uses:

- Pre-extracted audio features (total240, 240‑dim)
- Pre-computed OOF predictions from two frozen upstream models (High‑SR & Pairwise)
- All hyper‑parameters inlined as constants

**Anti‑leakage guarantee:** The selection lock is written to disk *before* any robot/test data is loaded."""))


# ──────────────────────── CELL 1 : Imports ───────────────────────────
cells.append(code(r'''# ============================================================
# IMPORTS  (standard library / PyPI only — no project imports)
# ============================================================
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

print("=" * 60)
print(f"Python     {sys.version}")
print(f"numpy      {np.__version__}")
print(f"pandas     {pd.__version__}")
print(f"joblib     {joblib.__version__}")
print(f"scikit-learn {__import__('sklearn').__version__}")
print("=" * 60)
'''))


# ──────────────────────── CELL 2 : Constants ─────────────────────────
cells.append(code(r'''# ============================================================
# CONSTANTS & HYPER-PARAMETERS
# ============================================================

# --- Labels ----------------------------------------------------------
LABELS          = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS  = np.asarray([1, 2, 3], dtype=np.int64)
ID2LABEL        = {0: "ambient", 1: "leaf", 2: "trunk", 3: "twig"}
LABEL_MAP       = {"ambient": 0, "leaf": 1, "trunk": 2, "twig": 3}
CLASS_NAMES     = ["ambient", "leaf", "trunk", "twig"]
PROBA_COLUMNS   = [f"proba_{ID2LABEL[i]}" for i in LABELS]

# --- Selection search space (overridden from entry-point config) -----
HIGHSR_WEIGHT_CANDIDATES = [0.65, 0.70, 0.75, 0.80]
GROUP_RULES              = ["sum_log_proba"]

# --- Selection-scoring formula weights (hard-coded) ------------------
SELECTION_MACRO_WEIGHT   = 0.65
SELECTION_CONTACT_WEIGHT = 0.25
SELECTION_BINARY_WEIGHT  = 0.10

# --- Stress view names -----------------------------------------------
STRESS_VIEWS = ("clean", "robot_mix", "bandlimit")

# --- Frozen upstream model selection (read from prior lock files) ----
OUTPUT_DIR = Path("outputs")
BENCH_DIR  = OUTPUT_DIR / "audio_feature_benchmarks"
RUN_SLUG   = "audio_log_consensus_pair_blend_select"
ROOT_PATH  = Path("tree_structures")

# Pre-extracted feature cache
TRAIN_FEATURE_DIR  = BENCH_DIR / "total240_trainval_select" / "features"
PAIRWISE_SPLIT_DIR = BENCH_DIR / "audio_pairwise_contact_stress_cv_select" / "splits"
HIGHSR_RUN_DIR     = BENCH_DIR / "audio_highsr_temporal_tta_select"
PAIRWISE_RUN_DIR   = BENCH_DIR / "audio_pairwise_contact_stress_cv_select"
RUN_DIR            = BENCH_DIR / RUN_SLUG
REPORT_DIR         = RUN_DIR / "reports"
MODEL_DIR          = RUN_DIR / "models"
OOF_SOURCES_DIR    = RUN_DIR / "oof_sources"

for d in [RUN_DIR, REPORT_DIR, MODEL_DIR, OOF_SOURCES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Read upstream high-SR lock ───────────────────────────────────────
_hsr_lock = json.loads(
    (HIGHSR_RUN_DIR / "reports"
     / "audio_highsr_temporal_tta_select_selected_without_test.json")
    .read_text()
)
HIGHSR_SELECTED_CANDIDATE = str(_hsr_lock["selected_candidate"])
HIGHSR_TTA_WEIGHTS        = _hsr_lock["selected_tta_weights"]
#   Example: {"clean": 0.6, "robot_mix": 0.4}
del _hsr_lock

# ── Read upstream pairwise lock ──────────────────────────────────────
_pair_lock = json.loads(
    (PAIRWISE_RUN_DIR / "reports"
     / "audio_pairwise_contact_stress_cv_select_selected_without_test.json")
    .read_text()
)
PAIRWISE_SELECTED_CANDIDATE = str(_pair_lock["selected_candidate"])
del _pair_lock

print("Constants initialized.")
print(f"  HIGHSR_WEIGHT_CANDIDATES = {HIGHSR_WEIGHT_CANDIDATES}")
print(f"  GROUP_RULES              = {GROUP_RULES}")
print(f"  High-SR candidate         = {HIGHSR_SELECTED_CANDIDATE}")
print(f"  High-SR TTA weights       = {HIGHSR_TTA_WEIGHTS}")
print(f"  Pairwise candidate        = {PAIRWISE_SELECTED_CANDIDATE}")
print(f"  Selection-score formula   = {SELECTION_MACRO_WEIGHT}*macro "
      f"+ {SELECTION_CONTACT_WEIGHT}*contact "
      f"+ {SELECTION_BINARY_WEIGHT}*binary")
print(f"  Output dir                = {RUN_DIR.resolve()}")
'''))


# ──────────────────────── CELL 3 : Utility Functions ─────────────────
cells.append(code(r'''# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize(proba: np.ndarray) -> np.ndarray:
    """Clip + L1-normalize probability vectors."""
    proba = np.clip(proba, 1e-12, 1.0)
    return proba / proba.sum(axis=1, keepdims=True)


def fast_macro_f1(y_true: np.ndarray, pred: np.ndarray,
                  labels: np.ndarray = LABELS) -> float:
    """Vectorized macro F1 — identical to the original script."""
    scores = []
    for label in labels:
        t = y_true == label
        p = pred   == label
        tp  = float(np.sum(t & p))
        fp  = float(np.sum(~t & p))
        fn  = float(np.sum(t & ~p))
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def segment_group_key(audio_file: str) -> str:
    """Strip trailing _window_<n>... from filename stem."""
    stem = Path(audio_file).stem
    return re.sub(r"_window_\d+.*$", "", stem)


def load_manifest(csv_path: Path, source: str) -> pd.DataFrame:
    """Load dataset.csv, validate, add computed columns."""
    frame = pd.read_csv(csv_path)
    required = {"audio_file", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{csv_path} missing columns: {sorted(required)}")

    base_dir = csv_path.parent
    output = pd.DataFrame({
        "audio_file": frame["audio_file"].astype(str),
        "image_file": frame.get("image_file",
                                pd.Series([""] * len(frame))).astype(str),
        "audio_path": frame["audio_file"].map(lambda v: base_dir / str(v)),
        "label":      frame["category"].astype(str).str.lower(),
        "source":     source,
    })
    output = output[output["label"].isin(LABEL_MAP)].copy()
    output["y"] = output["label"].map(LABEL_MAP).astype(np.int64)
    output["group_key"] = output["audio_file"].map(segment_group_key)
    output = output.reset_index(drop=True)

    missing = set(CLASS_NAMES) - set(output["label"].unique())
    if missing:
        raise ValueError(f"{source} missing classes: {sorted(missing)}")
    missing_paths = ~output["audio_path"].map(lambda p: Path(p).exists())
    if missing_paths.any():
        examples = output.loc[missing_paths, "audio_path"].head(5).tolist()
        raise FileNotFoundError(
            f"{source} has {int(missing_paths.sum())} missing: {examples}"
        )
    return output


def group_decode(frame: pd.DataFrame, proba: np.ndarray,
                 rule: str) -> np.ndarray:
    """Apply segment-level consensus decoding.

    - 'sum_log_proba': sum log-probabilities per group → argmax
    - 'mean_proba': average probabilities per group → argmax
    """
    group_codes, _ = pd.factorize(frame["group_key"].astype(str),
                                  sort=False)
    n_groups = int(group_codes.max()) + 1
    if rule == "sum_log_proba":
        values = np.log(np.clip(proba, 1e-12, 1.0))
    else:
        values = proba

    sums = np.vstack([
        np.bincount(group_codes, weights=values[:, c],
                    minlength=n_groups)
        for c in LABELS
    ]).T

    if rule == "mean_proba":
        counts = np.bincount(group_codes,
                             minlength=n_groups).astype(np.float64)
        sums = sums / counts[:, None]
    elif rule != "sum_log_proba":
        raise KeyError(f"Unknown group rule: {rule}")

    group_pred = np.argmax(sums, axis=1).astype(np.int64)
    return group_pred[group_codes]


def weighted_proba(proba_by_view: dict, weights: dict) -> np.ndarray:
    """Weighted blend of multi-view probabilities (TTA)."""
    total = float(sum(weights.values()))
    output = None
    for view, weight in weights.items():
        part = (float(weight) / total) * proba_by_view[view]
        output = part if output is None else output + part
    output = np.clip(output, 1e-12, 1.0)
    return output / output.sum(axis=1, keepdims=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float),
                    encoding="utf-8")


def label_counts(y: np.ndarray) -> dict:
    return {ID2LABEL[int(lbl)]: int(np.sum(y == lbl)) for lbl in LABELS}


print("All utility functions defined.")
'''))


# ──────────────────────── CELL 4 : Load Train Data ───────────────────
cells.append(code(r'''# ============================================================
# PHASE 1: LOAD TRAIN DATA  (hand/default only)
# ============================================================

print("Loading train manifest …")
train_csv = ROOT_PATH / "audio_visual_dataset_default" / "dataset.csv"
if not train_csv.exists():
    raise FileNotFoundError(f"Missing dataset: {train_csv}")
train_df = load_manifest(train_csv, "hand_train")
print(f"  {len(train_df)} rows, {train_df['group_key'].nunique()} groups")

# --- Pre-extracted features ---
t0 = time.perf_counter()
X_train = np.load(
    TRAIN_FEATURE_DIR / "hand_train_full" / "X.npy"
).astype(np.float32)
y_train = np.load(
    TRAIN_FEATURE_DIR / "hand_train_full" / "y.npy"
).astype(np.int64)
print(f"  Features loaded in {time.perf_counter() - t0:.3f}s")
print(f"  X_train = {X_train.shape}  y_train = {y_train.shape}")
print(f"  Label distribution: {label_counts(y_train)}")

# --- CV fold assignment (from pairwise experiment) ---
fold_path = (PAIRWISE_SPLIT_DIR /
             "hand_train_full_pairwise_contact_stress_cv_folds.csv")
fold_assignment = pd.read_csv(fold_path)["cv_fold"].to_numpy(dtype=np.int64)
print(f"  Fold counts: {dict(zip(*np.unique(fold_assignment, return_counts=True)))}")

# --- Sanity checks ---------------------------------------------------
assert len(train_df) == len(X_train) == len(y_train) == len(fold_assignment), \
    "Length mismatch between manifest, features, and folds!"
assert train_df["y"].equals(pd.Series(y_train)), \
    "Label mismatch between manifest and cached features!"
print("  Data aligned ✓")
'''))


# ──────────────────────── CELL 5 : Load OOF ──────────────────────────
cells.append(code(r'''# ============================================================
# PHASE 2: LOAD OOF PREDICTIONS
#          (train-only — from frozen upstream models)
# ============================================================

# --- High-SR OOF -----------------------------------------------------
print("Loading High-SR OOF …")
hsr_oof_dir = HIGHSR_RUN_DIR / "oof_proba" / HIGHSR_SELECTED_CANDIDATE
highsr_views: dict[str, np.ndarray] = {}
for view in STRESS_VIEWS:
    path = hsr_oof_dir / f"{view}_oof_proba.npy"
    highsr_views[view] = np.load(path)
    print(f"  {view}: {highsr_views[view].shape}")

highsr_oof = normalize(weighted_proba(highsr_views, HIGHSR_TTA_WEIGHTS))
print(f"  blended  →  {highsr_oof.shape}")

# --- Pairwise OOF ----------------------------------------------------
print("\nLoading Pairwise OOF …")
pairwise_oof_path = OOF_SOURCES_DIR / "pairwise_selected_clean_oof_proba.npy"
if not pairwise_oof_path.exists():
    raise FileNotFoundError(
        f"Pairwise OOF cache not found: {pairwise_oof_path}\n"
        "Run the original script with --force-rebuild-pairwise-oof first."
    )
pairwise_oof = normalize(np.load(pairwise_oof_path))
print(f"  {pairwise_oof.shape}")

# --- Verify OOF dimensions match train data --------------------------
assert highsr_oof.shape == (len(y_train), len(LABELS)), \
    f"High-SR OOF shape mismatch: {highsr_oof.shape}"
assert pairwise_oof.shape == (len(y_train), len(LABELS)), \
    f"Pairwise OOF shape mismatch: {pairwise_oof.shape}"
print("  OOF aligned ✓")
'''))


# ──────────────────────── CELL 6 : Evaluate Candidates ───────────────
cells.append(code(r'''# ============================================================
# PHASE 3: OOF SELECTION  (train-only — NO test data)
# ============================================================

print("Evaluating blend candidates on train OOF …\n")
print(f"  Weight grid:  {HIGHSR_WEIGHT_CANDIDATES}")
print(f"  Group rules:  {GROUP_RULES}")
print()

rows = []
for hw in HIGHSR_WEIGHT_CANDIDATES:
    pw = 1.0 - hw
    blended = normalize(hw * highsr_oof + pw * pairwise_oof)
    for rule in GROUP_RULES:
        pred    = group_decode(train_df, blended, rule)
        macro   = fast_macro_f1(y_train, pred, LABELS)
        contact = fast_macro_f1(y_train, pred, CONTACT_LABELS)
        binary  = fast_macro_f1((y_train > 0).astype(np.int64),
                                (pred > 0).astype(np.int64),
                                np.asarray([0, 1]))
        sel = (SELECTION_MACRO_WEIGHT   * macro
               + SELECTION_CONTACT_WEIGHT * contact
               + SELECTION_BINARY_WEIGHT  * binary)
        rows.append({
            "recipe_kind":       "audio_segment_consistency_pair_blend",
            "highsr_weight":     float(hw),
            "pairwise_weight":   float(pw),
            "group_rule":        rule,
            "macro_f1":          macro,
            "contact_macro_f1":  contact,
            "binary_macro_f1":   binary,
            "selection_score":   float(sel),
        })
        print(f"  w={hw:.2f}  pw={pw:.2f}  rule={rule:>14s}  "
              f"macro={macro:.4f}  contact={contact:.4f}  "
              f"binary={binary:.4f}  →  sel={sel:.4f}")

leaderboard = (
    pd.DataFrame(rows)
    .sort_values(["selection_score", "macro_f1", "contact_macro_f1"],
                 ascending=False)
    .reset_index(drop=True)
)

print(f"\n{'=' * 70}")
print("LEADERBOARD  (train OOF — sorted by selection_score)")
print(f"{'=' * 70}")
print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

selected = leaderboard.iloc[0].to_dict()
print(f"\n{'─' * 70}")
print(f"  ✓ SELECTED:  highsr_weight = {selected['highsr_weight']}")
print(f"               pairwise_weight = {selected['pairwise_weight']}")
print(f"               group_rule      = {selected['group_rule']}")
print(f"               OOF macro_f1    = {selected['macro_f1']:.4f}")
print(f"               selection_score = {selected['selection_score']:.4f}")
print(f"{'─' * 70}")
'''))


# ──────────────────────── CELL 7 : Selection Lock ────────────────────
cells.append(code(r'''# ============================================================
# PHASE 4: SELECTION LOCK  (write BEFORE test data is touched)
# ============================================================

method_card = {
    "protocol": "audio_only_segment_consistency_pair_blend_no_test_until_lock",
    "allowed_selection_data": (
        "hand/default train labels, train group_key, "
        "locked high-SR OOF, rebuilt pairwise OOF"
    ),
    "forbidden_selection_data": (
        "robot/test labels or robot/test predictions before selection lock; "
        "image/multimodal features"
    ),
    "candidate_count":          int(len(leaderboard)),
    "highsr_weight_candidates": HIGHSR_WEIGHT_CANDIDATES,
    "group_rules":              GROUP_RULES,
    "selection_formula": {
        "macro_weight":   SELECTION_MACRO_WEIGHT,
        "contact_weight": SELECTION_CONTACT_WEIGHT,
        "binary_weight":  SELECTION_BINARY_WEIGHT,
    },
    "label_counts":             label_counts(y_train),
    "train_groups":             int(train_df["group_key"].nunique()),
}

method_card_path = REPORT_DIR / f"{RUN_SLUG}_method_card_before_test.json"
leaderboard_path = REPORT_DIR / f"{RUN_SLUG}_oof_group_leaderboard.csv"
selection_path   = REPORT_DIR / f"{RUN_SLUG}_selected_without_test.json"

write_json(method_card_path, method_card)
leaderboard.to_csv(leaderboard_path, index=False)

selection_summary = {
    "selection_rule": (
        "best train-only OOF segment-consistency blend of "
        "high-SR and pairwise audio sources"
    ),
    "selected_without_test": selected,
    "method_card":           str(method_card_path.resolve()),
    "leaderboard_path":      str(leaderboard_path.resolve()),
}
write_json(selection_path, selection_summary)

print("=" * 60)
print("🔒  SELECTION LOCK  written BEFORE loading robot/test")
print("=" * 60)
print(json.dumps(selection_summary, indent=2, default=float))
print()
print("─" * 60)
print("  ║  NO test data accessed above this line  ║")
print("─" * 60)
'''))


# ──────────────────────── CELL 8 : Load Test Data ────────────────────
cells.append(code(r'''# ============================================================
# PHASE 5: LOAD TEST DATA  (robot/test — AFTER lock)
# ============================================================

def load_final_frame_and_proba(path: Path):
    """Read a final-test predictions CSV and return frame + proba matrix."""
    frame = pd.read_csv(path)
    proba_cols = [f"proba_{ID2LABEL[i]}" for i in range(4)]
    return frame, normalize(frame[proba_cols].to_numpy(dtype=np.float64))

highsr_frame, highsr_final = load_final_frame_and_proba(
    HIGHSR_RUN_DIR / "reports"
    / "audio_highsr_temporal_tta_select_final_test_predictions.csv"
)
pair_frame, pair_final = load_final_frame_and_proba(
    PAIRWISE_RUN_DIR / "reports"
    / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
)

print(f"High-SR test predictions:   {highsr_final.shape}")
print(f"Pairwise test predictions:  {pair_final.shape}")

# --- Verify source predictions are aligned ---------------------------
for col in ["audio_file", "y"]:
    assert np.array_equal(
        highsr_frame[col].astype(str).to_numpy(),
        pair_frame[col].astype(str).to_numpy(),
    ), f"Final source frames are not aligned on column '{col}'"

y_test = highsr_frame["y"].to_numpy(dtype=np.int64)
print(f"\nTest samples:          {len(y_test)}")
print(f"Test label distribution: {label_counts(y_test)}")
print("  Test data aligned ✓")
'''))


# ──────────────────────── CELL 9 : Final Blend & Predict ─────────────
cells.append(code(r'''# ============================================================
# PHASE 6: FINAL BLEND + GROUP DECODE  (frozen selection → robot/test)
# ============================================================

highsr_w   = float(selected["highsr_weight"])
pairwise_w = float(selected["pairwise_weight"])
rule       = str(selected["group_rule"])

print(f"Frozen config:  highsr_w = {highsr_w}")
print(f"                pairwise_w = {pairwise_w}")
print(f"                group_rule  = \"{rule}\"")
print()

# --- Blend -----------------------------------------------------------
final_blend = normalize(highsr_w * highsr_final + pairwise_w * pair_final)
print(f"  blended proba: {final_blend.shape} (min={final_blend.min():.6f}, max={final_blend.max():.6f})")

# --- Group decode ----------------------------------------------------
t0 = time.perf_counter()
final_pred = group_decode(highsr_frame, final_blend, rule)
print(f"  decode time: {time.perf_counter() - t0:.4f}s")

# --- Score -----------------------------------------------------------
cm = confusion_matrix(y_test, final_pred, labels=LABELS)
accuracy_4class = float(np.diag(cm).sum() / cm.sum())
macro_f1_4class = f1_score(y_test, final_pred, labels=LABELS,
                           average="macro", zero_division=0)
contact_macro   = f1_score(y_test, final_pred, labels=CONTACT_LABELS,
                           average="macro", zero_division=0)
y_bin_true      = (y_test > 0).astype(np.int64)
y_bin_pred      = (final_pred > 0).astype(np.int64)
binary_accuracy = float((y_bin_true == y_bin_pred).mean())
binary_macro    = f1_score(y_bin_true, y_bin_pred, labels=[0, 1],
                           average="macro", zero_division=0)

oof_gap = float(selected["macro_f1"]) - macro_f1_4class

print()
print("=" * 70)
print("  FINAL TEST RESULTS")
print("  (frozen audio segment-consistency blend, robot/test)")
print("=" * 70)
print(f"  accuracy_4class           = {accuracy_4class:.6f}")
print(f"  macro_f1_4class           = {macro_f1_4class:.6f}")
print(f"  contact_macro_f1          = {contact_macro:.6f}")
print(f"  binary_macro_f1           = {binary_macro:.6f}")
print(f"  binary_accuracy           = {binary_accuracy:.6f}")
print()
print(f"  selected_oof_macro_f1     = {float(selected['macro_f1']):.6f}")
print(f"  selected_highsr_weight    = {highsr_w}")
print(f"  selected_pairwise_weight  = {pairwise_w}")
print(f"  selected_group_rule       = {rule}")
print(f"  OOF → test gap (macro F1) = {float(selected['macro_f1']):.4f} → {macro_f1_4class:.4f}  (Δ={oof_gap:+.4f})")
print()
print("  Confusion Matrix:")
print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
'''))


# ──────────────────────── CELL 10 : Per-Class ────────────────────────
cells.append(code(r'''# ============================================================
# PER-CLASS BREAKDOWN
# ============================================================

from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, final_pred, target_names=CLASS_NAMES,
                            digits=4, zero_division=0))

per_class = []
for i, name in enumerate(CLASS_NAMES):
    t   = y_test == i
    p   = final_pred == i
    tp  = float(np.sum(t & p))
    fp  = float(np.sum(~t & p))
    fn  = float(np.sum(t & ~p))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1v  = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    supp = int(np.sum(t))
    per_class.append({
        "Class": name, "Precision": prec,
        "Recall": rec, "F1": f1v, "Support": supp,
    })

print()
print(pd.DataFrame(per_class).to_string(index=False, float_format="%.4f"))
'''))


# ──────────────────────── CELL 11 : Artifacts ────────────────────────
cells.append(code(r'''# ============================================================
# PHASE 7: SAVE ARTIFACTS
# ============================================================

final_report_path = REPORT_DIR / f"{RUN_SLUG}_final_test_report.csv"
predictions_path  = REPORT_DIR / f"{RUN_SLUG}_final_test_predictions.csv"
confusion_path    = REPORT_DIR / f"{RUN_SLUG}_final_test_confusion_matrix.csv"
bundle_path       = MODEL_DIR  / f"{RUN_SLUG}_selected_model_bundle.joblib"
summary_path      = REPORT_DIR / f"{RUN_SLUG}_protocol_summary.json"

# 1. Final test report (1-row CSV)
final_row = {
    "feature_set":                "total240",
    "feature_name":               "Total 240D",
    "n_features":                 240,
    "split":                      "robot_test_final",
    "model":        "audio_group_consistency_highsr_pairwise_blend",
    "status":                     "ok",
    "accuracy_4class":            accuracy_4class,
    "macro_f1_4class":            macro_f1_4class,
    "contact_macro_f1":           contact_macro,
    "binary_macro_f1":            binary_macro,
    "binary_accuracy":            binary_accuracy,
    "selected_by": "train_only_oof_audio_segment_consistency",
    "selected_score":             selected["selection_score"],
    "selected_oof_macro_f1":      selected["macro_f1"],
    "selected_oof_contact_macro_f1": selected["contact_macro_f1"],
    "selected_highsr_weight":     highs_rw,
    "selected_pairwise_weight":   pairwise_w,
    "selected_group_rule":        rule,
    "oof_test_macro_gap":         oof_gap,
}
pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
print(f"[ok] {final_report_path}")

# 2. Predictions CSV
pred_cols = [c for c in ["audio_file", "label", "y", "group_key", "source"]
             if c in highsr_frame.columns]
pred_frame = highsr_frame[pred_cols].copy()
pred_frame["pred_y"]     = final_pred.astype(int)
pred_frame["pred_label"] = pred_frame["pred_y"].map(ID2LABEL)
for ci, cn in ID2LABEL.items():
    pred_frame[f"proba_{cn}"] = final_blend[:, ci]
pred_frame.to_csv(predictions_path, index=False)
print(f"[ok] {predictions_path}")

# 3. Confusion matrix
pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(confusion_path)
print(f"[ok] {confusion_path}")

# 4. Model bundle
joblib.dump(
    {
        "protocol":           method_card["protocol"],
        "method_card":        method_card,
        "selection_summary":  selection_summary,
        "final_test_report":  final_row,
        "config": {
            "highsr_weight_candidates":    HIGHSR_WEIGHT_CANDIDATES,
            "group_rules":                 GROUP_RULES,
            "highsr_selected_candidate":   HIGHSR_SELECTED_CANDIDATE,
            "highsr_tta_weights":          HIGHSR_TTA_WEIGHTS,
            "pairwise_selected_candidate": PAIRWISE_SELECTED_CANDIDATE,
            "selection_formula": {
                "macro":   SELECTION_MACRO_WEIGHT,
                "contact": SELECTION_CONTACT_WEIGHT,
                "binary":  SELECTION_BINARY_WEIGHT,
            },
        },
    },
    bundle_path,
)
print(f"[ok] {bundle_path}")

# 5. Protocol summary
protocol_summary = {
    "protocol":          method_card["protocol"],
    "root_path":         str(ROOT_PATH.resolve()),
    "run_dir":           str(RUN_DIR.resolve()),
    "method_card":       method_card,
    "selection_summary": selection_summary,
    "final_test_report": final_row,
    "artifacts": {
        "method_card":               str(method_card_path.resolve()),
        "leaderboard":               str(leaderboard_path.resolve()),
        "selection_lock":            str(selection_path.resolve()),
        "final_test_report":         str(final_report_path.resolve()),
        "final_test_predictions":    str(predictions_path.resolve()),
        "final_test_confusion_matrix": str(confusion_path.resolve()),
        "selected_model_bundle":     str(bundle_path.resolve()),
    },
}
write_json(summary_path, protocol_summary)
print(f"[ok] {summary_path}")

print("\n✓ All artifacts saved to:")
print(f"    {REPORT_DIR.resolve()}/")
print(f"    {MODEL_DIR.resolve()}/")
'''))


# ──────────────────────── CELL 12 : Summary ──────────────────────────
cells.append(code(r'''# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print(f"  PROTOCOL:  {method_card['protocol']}")
print("=" * 70)

print(f"""
Selection (train OOF only, locked BEFORE test):
  Candidates evaluated:    {len(leaderboard)}
  Best blend weight:       highsr = {highsr_w}  |  pairwise = {pairwise_w}
  Group rule:              {rule}
  OOF macro_f1:            {float(selected['macro_f1']):.4f}
  OOF contact_macro_f1:    {float(selected['contact_macro_f1']):.4f}
  OOF binary_macro_f1:     {float(selected['binary_macro_f1']):.4f}
  Selection score:         {selected['selection_score']:.4f}

Test (robot/test, frozen config):
  Samples:                 {len(y_test)}
  Accuracy (4-class):      {accuracy_4class:.4f}
  Macro F1 (4-class):      {macro_f1_4class:.4f}
  Contact Macro F1:        {contact_macro:.4f}
  Binary Macro F1:         {binary_macro:.4f}
  OOF → test gap:          {float(selected['macro_f1']):.4f} → {macro_f1_4class:.4f}  (Δ={oof_gap:+.4f})

Verification:
  ✓ Selection lock written BEFORE any test data loaded
  ✓ Audio-only — no image/vision features
  ✓ Self-contained — zero project imports
  ✓ All artifacts saved to disk
""")

# Numerical cross-check against original run
print("─" * 70)
print("Cross-check against original output:")
print(f"  accuracy   notebook={accuracy_4class:.6f}  original=0.764759  match={abs(accuracy_4class - 0.764759) < 1e-5}")
print(f"  macro_f1   notebook={macro_f1_4class:.6f}  original=0.653825  match={abs(macro_f1_4class - 0.653825) < 1e-5}")
print(f"  contact_f1 notebook={contact_macro:.6f}  original=0.559534  match={abs(contact_macro - 0.559534) < 1e-5}")
print(f"  binary_f1  notebook={binary_macro:.6f}  original=0.930497  match={abs(binary_macro - 0.930497) < 1e-5}")
print("─" * 70)
'''))


# ──────────────────────── CELL 13 : Feature Extraction (reference) ───
cells.append(md("""## Appendix: Feature Extraction Code

The features used by the upstream models (total240, 240‑dimensional) are
extracted with the following functions. This code is included for
reference — the notebook above uses pre‑extracted `.npy` caches and does
not call these extraction functions directly."""))

cells.append(code(r'''# ============================================================
# APPENDIX: AUDIO FEATURE EXTRACTION  (reference — not executed)
# ============================================================
# The upstream high-SR and pairwise models extract features via
# these functions.  The notebook loads pre-extracted .npy caches
# and does NOT re-extract.

import librosa

CONFIG = {
    "audio": {"sr": 16000, "duration": 1.0},
    "event_crop": {"top_energy_sec": 0.4, "top_energy_hop_sec": 0.05},
    "stft_base": {"n_fft": 512, "win_length": 400, "hop_length": 160, "window": "hann"},
    "mfcc": {"n_mfcc": 20, "n_mels": 64, "fmin": 20, "fmax": 8000},
    "mel": {"n_mels": 64, "n_groups": 7},
    "fft": {"bands": [(20,100),(100,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,6000),(6000,8000)]},
    "class_weights": {0: 0.6, 1: 1.1, 2: 1.8, 3: 1.3},
}

def load_audio(path, target_sr=16000, duration=1.0):
    signal, _ = librosa.load(path, sr=target_sr, mono=True)
    target_len = int(target_sr * duration)
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    else:
        signal = signal[:target_len]
    peak = np.max(np.abs(signal)) + 1e-8
    return (signal / peak).astype(np.float32)

def get_top_energy_window(signal, sr=16000, window_sec=0.4, hop_sec=0.05):
    wl, hl = int(window_sec * sr), int(hop_sec * sr)
    if len(signal) <= wl: return signal
    best_start, best_energy = 0, -1.0
    for start in range(0, len(signal) - wl + 1, hl):
        energy = float(np.mean(signal[start:start+wl]**2))
        if energy > best_energy:
            best_energy, best_start = energy, start
    return signal[best_start:best_start+wl]

def summarize_4(values):
    v = np.asarray(values)
    return np.asarray([np.mean(v), np.std(v), np.max(v), np.percentile(v, 90)], dtype=np.float32)

def extract_mfcc_compact(signal, sr=16000):
    sc = CONFIG["stft_base"]; mc = CONFIG["mfcc"]
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=mc["n_mfcc"], n_mels=mc["n_mels"],
                                n_fft=sc["n_fft"], win_length=sc["win_length"],
                                hop_length=sc["hop_length"], fmin=mc["fmin"], fmax=mc["fmax"])
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).astype(np.float32)

def extract_stft_compact(signal, sr=16000):
    sc = CONFIG["stft_base"]
    mag = np.abs(librosa.stft(signal, n_fft=sc["n_fft"], hop_length=sc["hop_length"],
                              win_length=sc["win_length"], window=sc["window"]))
    rms = librosa.feature.rms(S=mag, frame_length=sc["n_fft"], hop_length=sc["hop_length"])[0]
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=sc["n_fft"], hop_length=sc["hop_length"])[0]
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=mag, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=mag)[0]
    norm = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-8)
    flux = np.sqrt(np.sum(np.diff(norm, axis=1)**2, axis=0))
    flux = np.pad(flux, (1, 0))
    return np.concatenate([summarize_4(s) for s in [rms, zcr, centroid, bandwidth, rolloff, flatness, flux]]).astype(np.float32)

def extract_mel_compact(signal, sr=16000):
    sc = CONFIG["stft_base"]; mc = CONFIG["mfcc"]; ml = CONFIG["mel"]
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=ml["n_mels"],
                                         n_fft=sc["n_fft"], win_length=sc["win_length"],
                                         hop_length=sc["hop_length"],
                                         fmin=mc["fmin"], fmax=mc["fmax"], power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    parts = [summarize_4(np.mean(g, axis=0)) for g in np.array_split(log_mel, ml["n_groups"], axis=0)]
    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts]).astype(np.float32)

def extract_fft_compact(signal, sr=16000):
    eps = 1e-8
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0/sr)
    power = fft**2; total = np.sum(power) + eps
    band_powers = []
    for lo, hi in CONFIG["fft"]["bands"]:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        band_powers.append(np.sum(power[idx]) / total)
    bp = np.asarray(band_powers, dtype=np.float32)
    log_bp = np.log1p(bp)
    low, mid, high = np.sum(bp[0:3]), np.sum(bp[3:6]), np.sum(bp[6:8])
    ratios = np.asarray([low/(mid+eps), low/(high+eps), mid/(high+eps), (low+mid)/(high+eps)], dtype=np.float32)
    prob = power / total
    entropy = -np.sum(prob * np.log(prob + eps)) / np.log(len(prob))
    centroid = np.sum(freqs * power) / total
    dominant = freqs[int(np.argmax(power))]
    extras = np.asarray([entropy, centroid, dominant, fft[int(np.argmax(power))]], dtype=np.float32)
    return np.concatenate([bp, log_bp, ratios, extras]).astype(np.float32)

def extract_total_120(signal, sr=16000):
    return np.concatenate([
        extract_mfcc_compact(signal, sr),
        extract_stft_compact(signal, sr),
        extract_mel_compact(signal, sr),
        extract_fft_compact(signal, sr),
    ]).astype(np.float32)

def extract_total_240(signal, sr=16000):
    top = get_top_energy_window(signal, sr, window_sec=0.4, hop_sec=0.05)
    return np.concatenate([
        extract_total_120(signal, sr),
        extract_total_120(top, sr),
    ]).astype(np.float32)

print("Feature extraction reference code — not executed.")
print("Notebook uses pre-extracted .npy caches from:")
print(f"  {TRAIN_FEATURE_DIR.resolve()}")
'''))


# ────────── build & write notebook ───────────────────────────────────
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0",
    },
}

# Validate
nbf.validate(nb)

OUTPUT_PATH.write_text(nbf.writes(nb), encoding="utf-8")
print(f"✓ Notebook written: {OUTPUT_PATH}  ({os.path.getsize(OUTPUT_PATH):,} bytes)")
''')
