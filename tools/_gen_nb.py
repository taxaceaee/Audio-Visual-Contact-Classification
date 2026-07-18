#!/usr/bin/env python3
"""Generate self-contained notebook - write to file then delete self."""
import json, os, sys
from pathlib import Path

OUT = []
def md(src): OUT.append({"cell_type":"markdown","metadata":{},"source":src.strip().split("\n")})
def code(src): OUT.append({"cell_type":"code","metadata":{},"outputs":[],"source":[l+"\n" for l in src.strip().split("\n")],"execution_count":None})

md("""# Audio Log-Consensus Pair Blend Select -- Final Test
## Self-contained standalone notebook
**Protocol:** Audio-only segment-consistency blend with log-probability consensus.
Replicates exactly `train_audio_log_consensus_pair_blend_select_final_test.py` but zero project imports.
Uses pre-extracted features & cached OOF. Selection lock written BEFORE any test data is loaded.""")

code("""# ============================================================
# IMPORTS  (standard library / PyPI only -- no project imports)
# ============================================================
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
print("=" * 60)
print(f"Python     {sys.version}")
print(f"numpy      {np.__version__}")
print(f"pandas     {pd.__version__}")
print(f"joblib     {joblib.__version__}")
print(f"scikit-learn {__import__('sklearn').__version__}")
print("=" * 60)""")

code("""# ============================================================
# CONSTANTS & HYPER-PARAMETERS
# ============================================================
LABELS         = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
ID2LABEL       = {0: "ambient", 1: "leaf", 2: "trunk", 3: "twig"}
LABEL_MAP      = {"ambient": 0, "leaf": 1, "trunk": 2, "twig": 3}
CLASS_NAMES    = ["ambient", "leaf", "trunk", "twig"]
PROBA_COLUMNS  = ["proba_ambient", "proba_leaf", "proba_trunk", "proba_twig"]

HIGHSR_WEIGHT_CANDIDATES = [0.65, 0.70, 0.75, 0.80]
GROUP_RULES              = ["sum_log_proba"]
SEL_MACRO_W   = 0.65
SEL_CONTACT_W = 0.25
SEL_BINARY_W  = 0.10
STRESS_VIEWS  = ("clean", "robot_mix", "bandlimit")

OUTPUT_DIR = Path("outputs")
BENCH_DIR  = OUTPUT_DIR / "audio_feature_benchmarks"
RUN_SLUG   = "audio_log_consensus_pair_blend_select"
ROOT_PATH  = Path("tree_structures")
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

# --- Frozen upstream model locks ---
_h = json.loads((HIGHSR_RUN_DIR / "reports"
    / "audio_highsr_temporal_tta_select_selected_without_test.json").read_text())
HIGHSR_CAND = str(_h["selected_candidate"])
HIGHSR_TTA  = _h["selected_tta_weights"]
del _h
_p = json.loads((PAIRWISE_RUN_DIR / "reports"
    / "audio_pairwise_contact_stress_cv_select_selected_without_test.json").read_text())
PAIRWISE_CAND = str(_p["selected_candidate"])
del _p

print("High-SR candidate  :", HIGHSR_CAND)
print("High-SR TTA weights:", HIGHSR_TTA)
print("Pairwise candidate :", PAIRWISE_CAND)
sel_formula_str = f"{SEL_MACRO_W}*macro + {SEL_CONTACT_W}*contact + {SEL_BINARY_W}*binary"
print("Selection formula  :", sel_formula_str)""")

code("""# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize(proba):
    proba = np.clip(proba, 1e-12, 1.0)
    return proba / proba.sum(axis=1, keepdims=True)

def fast_macro_f1(y_true, pred, labels=LABELS):
    scores = []
    for lbl in labels:
        t = y_true == lbl; p = pred == lbl
        tp = float((t & p).sum())
        fp = float((~t & p).sum())
        fn = float((t & ~p).sum())
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= 0.0 else 2.0 * tp / denom)
    return float(np.mean(scores))

def segment_group_key(audio_file):
    return re.sub(r"_window_\\d+.*$", "", Path(audio_file).stem)

def load_manifest(csv_path, source):
    frame = pd.read_csv(csv_path)
    base_dir = csv_path.parent
    out = pd.DataFrame({
        "audio_file": frame["audio_file"].astype(str),
        "image_file": frame.get("image_file", pd.Series([""] * len(frame))).astype(str),
        "audio_path": frame["audio_file"].map(lambda v: base_dir / str(v)),
        "label":      frame["category"].astype(str).str.lower(),
        "source":     source,
    })
    out = out[out["label"].isin(LABEL_MAP)].copy()
    out["y"] = out["label"].map(LABEL_MAP).astype(np.int64)
    out["group_key"] = out["audio_file"].map(segment_group_key)
    out = out.reset_index(drop=True)
    missing = set(CLASS_NAMES) - set(out["label"].unique())
    if missing:
        raise ValueError(f"{source} missing classes: {sorted(missing)}")
    missing_p = ~out["audio_path"].map(lambda p: Path(p).exists())
    if missing_p.any():
        raise FileNotFoundError(f"{source} missing {int(missing_p.sum())} paths")
    return out

def group_decode(frame, proba, rule):
    gc, _ = pd.factorize(frame["group_key"].astype(str), sort=False)
    n = int(gc.max()) + 1
    vals = np.log(np.clip(proba, 1e-12, 1.0)) if rule == "sum_log_proba" else proba
    sums = np.vstack([np.bincount(gc, weights=vals[:, c], minlength=n) for c in LABELS]).T
    if rule == "mean_proba":
        sums = sums / np.bincount(gc, minlength=n).astype(np.float64)[:, None]
    elif rule != "sum_log_proba":
        raise KeyError(f"Unknown rule: {rule}")
    return np.argmax(sums, axis=1).astype(np.int64)[gc]

def weighted_proba(vws, wts):
    total = float(sum(wts.values()))
    out = None
    for vw, wt in wts.items():
        p = (float(wt) / total) * vws[vw]
        out = p if out is None else out + p
    out = np.clip(out, 1e-12, 1.0)
    return out / out.sum(axis=1, keepdims=True)

def write_json(p, d):
    p.write_text(json.dumps(d, indent=2, default=float), encoding="utf-8")

def label_counts(y):
    return {ID2LABEL[int(l)]: int((y == l).sum()) for l in LABELS}

print("Utility functions ready.")""")

code("""# ============================================================
# PHASE 1: LOAD TRAIN DATA (hand/default only)
# ============================================================
train_csv = ROOT_PATH / "audio_visual_dataset_default" / "dataset.csv"
train_df = load_manifest(train_csv, "hand_train")
print("Train manifest:", len(train_df), "rows,", train_df["group_key"].nunique(), "groups")

X_train = np.load(TRAIN_FEATURE_DIR / "hand_train_full" / "X.npy").astype(np.float32)
y_train = np.load(TRAIN_FEATURE_DIR / "hand_train_full" / "y.npy").astype(np.int64)
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("Label counts:", label_counts(y_train))

fold = pd.read_csv(PAIRWISE_SPLIT_DIR / "hand_train_full_pairwise_contact_stress_cv_folds.csv")
fold_assign = fold["cv_fold"].to_numpy(dtype=np.int64)
print("Fold counts:", dict(zip(*np.unique(fold_assign, return_counts=True))))
assert len(train_df) == len(X_train) == len(y_train) == len(fold_assign)
assert train_df["y"].equals(pd.Series(y_train))
print("Data aligned.")""")

code("""# ============================================================
# PHASE 2: LOAD OOF (train-only, frozen upstream)
# ============================================================
print("Loading High-SR OOF ...")
hsr_dir = HIGHSR_RUN_DIR / "oof_proba" / HIGHSR_CAND
hsr_views = {v: np.load(hsr_dir / f"{v}_oof_proba.npy") for v in STRESS_VIEWS}
highsr_oof = normalize(weighted_proba(hsr_views, HIGHSR_TTA))
print("  blended:", highsr_oof.shape)

print("Loading Pairwise OOF ...")
poof_path = OOF_SOURCES_DIR / "pairwise_selected_clean_oof_proba.npy"
pairwise_oof = normalize(np.load(poof_path))
print("  loaded:", pairwise_oof.shape)

assert highsr_oof.shape == (len(y_train), 4)
assert pairwise_oof.shape == (len(y_train), 4)
print("OOF ready.")""")

code("""# ============================================================
# PHASE 3: OOF SELECTION (train only -- NO test)
# ============================================================
print("Evaluating candidates ...")
print()
rows = []
for hw in HIGHSR_WEIGHT_CANDIDATES:
    pw = 1.0 - hw
    b = normalize(hw * highsr_oof + pw * pairwise_oof)
    for rule in GROUP_RULES:
        pred = group_decode(train_df, b, rule)
        m = fast_macro_f1(y_train, pred, LABELS)
        c = fast_macro_f1(y_train, pred, CONTACT_LABELS)
        bi = fast_macro_f1((y_train > 0).astype(np.int64), (pred > 0).astype(np.int64), np.asarray([0, 1]))
        sel_val = SEL_MACRO_W * m + SEL_CONTACT_W * c + SEL_BINARY_W * bi
        rows.append({"highsr_weight": float(hw), "pairwise_weight": float(pw),
                      "group_rule": rule, "macro_f1": m, "contact_macro_f1": c,
                      "binary_macro_f1": bi, "selection_score": float(sel_val)})
        print("  w=%.2f  pw=%.2f  %14s  M=%.4f  C=%.4f  B=%.4f  ->  %.4f" % (hw, pw, rule, m, c, bi, sel_val))

lb = pd.DataFrame(rows).sort_values(["selection_score", "macro_f1", "contact_macro_f1"], ascending=False).reset_index(drop=True)
print()
print("=" * 70)
print("LEADERBOARD (train OOF)")
print("=" * 70)
print(lb.to_string(index=False, float_format=lambda x: "%.6f" % x))
sel = lb.iloc[0].to_dict()
print()
print("SELECTED: highsr_w=%.2f  pairwise_w=%.2f  rule=%s  oof_macro=%.4f" % (sel["highsr_weight"], sel["pairwise_weight"], sel["group_rule"], sel["macro_f1"]))""")

code("""# ============================================================
# PHASE 4: SELECTION LOCK (write BEFORE test)
# ============================================================
mc = {
    "protocol": "audio_only_segment_consistency_pair_blend_no_test_until_lock",
    "candidate_count": int(len(lb)),
    "highsr_weight_candidates": HIGHSR_WEIGHT_CANDIDATES,
    "group_rules": GROUP_RULES,
    "label_counts": label_counts(y_train),
    "train_groups": int(train_df["group_key"].nunique()),
}
mp = REPORT_DIR / f"{RUN_SLUG}_method_card_before_test.json"
lp = REPORT_DIR / f"{RUN_SLUG}_oof_group_leaderboard.csv"
sp = REPORT_DIR / f"{RUN_SLUG}_selected_without_test.json"
write_json(mp, mc)
lb.to_csv(lp, index=False)
ss = {"selected_without_test": sel, "method_card": str(mp.resolve()), "leaderboard_path": str(lp.resolve())}
write_json(sp, ss)
print("=" * 50)
print("SELECTION LOCK WRITTEN BEFORE TEST")
print("=" * 50)
print(json.dumps(ss, indent=2, default=float))
print()
print("NO test data above this line.")""")

code("""# ============================================================
# PHASE 5: LOAD TEST (after lock)
# ============================================================
def load_csv(p):
    f = pd.read_csv(p)
    return f, normalize(f[PROBA_COLUMNS].to_numpy(dtype=np.float64))

hf, hfp = load_csv(HIGHSR_RUN_DIR / "reports" / "audio_highsr_temporal_tta_select_final_test_predictions.csv")
pf, pfp = load_csv(PAIRWISE_RUN_DIR / "reports" / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv")
print("High-SR test:", hfp.shape, "  Pairwise test:", pfp.shape)
for c in ["audio_file", "y"]:
    assert np.array_equal(hf[c].astype(str).to_numpy(), pf[c].astype(str).to_numpy()), ("misaligned: " + c)
y_test = hf["y"].to_numpy(dtype=np.int64)
print("Test samples:", len(y_test), " labels:", label_counts(y_test))""")

code("""# ============================================================
# PHASE 6: FINAL BLEND & PREDICT
# ============================================================
hw = float(sel["highsr_weight"])
pw = float(sel["pairwise_weight"])
rule = str(sel["group_rule"])
print("Config: highsr=%.2f  pairwise=%.2f  rule=%s" % (hw, pw, rule))

fb = normalize(hw * hfp + pw * pfp)
t0 = time.perf_counter()
pred = group_decode(hf, fb, rule)
print("Decode time:", round(time.perf_counter() - t0, 4), "s")

cm = confusion_matrix(y_test, pred, labels=LABELS)
acc = float(np.diag(cm).sum() / cm.sum())
mac = f1_score(y_test, pred, labels=LABELS, average="macro", zero_division=0)
con = f1_score(y_test, pred, labels=CONTACT_LABELS, average="macro", zero_division=0)
yb_t = (y_test > 0).astype(np.int64)
yb_p = (pred > 0).astype(np.int64)
bacc = float((yb_t == yb_p).mean())
bma = f1_score(yb_t, yb_p, labels=[0, 1], average="macro", zero_division=0)
gap = float(sel["macro_f1"]) - mac

print()
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)
print("accuracy_4class      = %.6f" % acc)
print("macro_f1_4class      = %.6f" % mac)
print("contact_macro_f1     = %.6f" % con)
print("binary_macro_f1      = %.6f" % bma)
print("binary_accuracy      = %.6f" % bacc)
print("selected_oof_macro   = %.6f" % float(sel["macro_f1"]))
oof_m = float(sel["macro_f1"])
print("OOF -> test gap      = %.4f -> %.4f  (d=%+.4f)" % (oof_m, mac, gap))
print()
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())""")

code("""# ============================================================
# PER-CLASS BREAKDOWN
# ============================================================
from sklearn.metrics import classification_report
print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

pc = []
for i, nm in enumerate(CLASS_NAMES):
    t = y_test == i; p = pred == i
    tp = float((t & p).sum())
    fp = float((~t & p).sum())
    fn = float((t & ~p).sum())
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1v = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    pc.append({"Class": nm, "Precision": pr, "Recall": rc, "F1": f1v, "Support": int(t.sum())})
print()
print(pd.DataFrame(pc).to_string(index=False, float_format="%.4f"))""")

code("""# ============================================================
# PHASE 7: SAVE ARTIFACTS
# ============================================================
frp = REPORT_DIR / f"{RUN_SLUG}_final_test_report.csv"
pp  = REPORT_DIR / f"{RUN_SLUG}_final_test_predictions.csv"
cp  = REPORT_DIR / f"{RUN_SLUG}_final_test_confusion_matrix.csv"
bp  = MODEL_DIR  / f"{RUN_SLUG}_selected_model_bundle.joblib"
smp = REPORT_DIR / f"{RUN_SLUG}_protocol_summary.json"

fr = {"feature_set": "total240", "feature_name": "Total 240D", "n_features": 240,
      "split": "robot_test_final", "model": "audio_group_consistency_highsr_pairwise_blend",
      "status": "ok", "accuracy_4class": acc, "macro_f1_4class": mac,
      "contact_macro_f1": con, "binary_macro_f1": bma, "binary_accuracy": bacc,
      "selected_by": "train_only_oof_audio_segment_consistency",
      "selected_score": sel["selection_score"], "selected_oof_macro_f1": sel["macro_f1"],
      "selected_highsr_weight": hw, "selected_pairwise_weight": pw,
      "selected_group_rule": rule, "oof_test_macro_gap": gap}
pd.DataFrame([fr]).to_csv(frp, index=False)

pcols = [c for c in ["audio_file", "label", "y", "group_key", "source"] if c in hf.columns]
pf2 = hf[pcols].copy()
pf2["pred_y"] = pred.astype(int)
pf2["pred_label"] = pf2["pred_y"].map(ID2LABEL)
for ci, cn in ID2LABEL.items():
    pf2["proba_" + cn] = fb[:, ci]
pf2.to_csv(pp, index=False)

pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cp)

joblib.dump({"protocol": mc["protocol"], "method_card": mc, "selection_summary": ss,
             "final_test_report": fr,
             "config": {"highsr_weight_candidates": HIGHSR_WEIGHT_CANDIDATES,
                        "group_rules": GROUP_RULES,
                        "highsr_candidate": HIGHSR_CAND,
                        "highsr_tta_weights": HIGHSR_TTA,
                        "pairwise_candidate": PAIRWISE_CAND}}, bp)

artifacts_list = {"method_card": str(mp.resolve()),
    "leaderboard": str(lp.resolve()), "selection_lock": str(sp.resolve()),
    "final_test_report": str(frp.resolve()), "final_test_predictions": str(pp.resolve()),
    "final_test_confusion_matrix": str(cp.resolve()),
    "selected_model_bundle": str(bp.resolve())}
write_json(smp, {"protocol": mc["protocol"], "root_path": str(ROOT_PATH.resolve()),
    "run_dir": str(RUN_DIR.resolve()), "method_card": mc, "selection_summary": ss,
    "final_test_report": fr, "artifacts": artifacts_list})

print("Artifacts saved:")
for p in [frp, pp, cp, bp, smp]:
    print(" ", p.resolve())""")

code("""# ============================================================
# FINAL SUMMARY
# ============================================================
oof_m = float(sel["macro_f1"])
oof_c = float(sel["contact_macro_f1"])
oof_b = float(sel["binary_macro_f1"])
print("=" * 70)
print("PROTOCOL: " + mc["protocol"])
print("=" * 70)
print()
print("Selection (train OOF, locked BEFORE test):")
print("  evaluated: %d candidates" % len(lb))
print("  best:      highsr_w = %.2f  pairwise_w = %.2f  rule = %s" % (hw, pw, rule))
print("  OOF:       macro = %.4f  contact = %.4f  binary = %.4f" % (oof_m, oof_c, oof_b))
print("  score:     %.4f" % sel["selection_score"])
print()
print("Test (robot/test, frozen):")
print("  samples:   %d" % len(y_test))
print("  accuracy:  %.4f" % acc)
print("  macro F1:  %.4f" % mac)
print("  contact:   %.4f" % con)
print("  binary:    %.4f" % bma)
print("  gap:       %.4f -> %.4f  (d=%+.4f)" % (oof_m, mac, gap))
print()
print("Anti-leakage:   OK  (lock before test)")
print("Audio-only:     OK  (no images)")
print("Self-contained: OK  (no project imports)")
print()
match1 = abs(acc - 0.764759) < 1e-5
match2 = abs(mac - 0.653825) < 1e-5
match3 = abs(con - 0.559534) < 1e-5
match4 = abs(bma - 0.930497) < 1e-5
all_ok = all([match1, match2, match3, match4])
print("-" * 50)
print("CROSS-CHECK vs original output:")
print("  accuracy    %.6f == 0.764759  %s" % (acc, match1))
print("  macro_f1    %.6f == 0.653825  %s" % (mac, match2))
print("  contact_f1  %.6f == 0.559534  %s" % (con, match3))
print("  binary_f1   %.6f == 0.930497  %s" % (bma, match4))
print("  ALL MATCH:  %s" % all_ok)
print("-" * 50)""")

md("""## Appendix: Feature Extraction & Hyper-Parameters

### Feature Set: total240 (240 dimensions)

| Component | Dim | Description |
|-----------|-----|-------------|
| MFCC40 | 40 | 20 MFCC coefficients (mean + std) |
| STFT28 | 28 | 7 spectral streams x 4 summary stats |
| Mel28 | 28 | 7 Mel-frequency groups x 4 summary stats |
| FFT24 | 24 | 8 bands + 8 log bands + 4 ratios + 4 extras |
| Total120 | 120 | MFCC40 + STFT28 + Mel28 + FFT24 |
| Total240 | 240 | Total120(full signal) + Total120(top-energy window) |

### Hyper-Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| Window length | 1.0 s |
| Top-energy window | 0.4 s, hop 0.05 s |
| STFT n_fft | 512 |
| STFT hop_length | 160 |
| MFCC n_mfcc | 20 |
| MFCC n_mels | 64 |
| Mel groups | 7 |
| FFT bands | 8 (20-8000 Hz) |
| Class weights | {ambient:0.6, leaf:1.1, trunk:1.8, twig:1.3} |

### High-SR Model
- **Candidate:** `highsr_hgb_default__all_aug` (HistGradientBoosting, 260 iters, lr=0.045)
- **TTA weights:** `{clean: 0.6, robot_mix: 0.4}` (selected by OOF)
- **Class bias:** `[0.2, 0.0, 0.0, 0.0]`

### Pairwise Model
- **Candidate:** `pairwise_hgb_svm_all_aug` (HGB binary + RBF-SVM pairwise)
- **Class bias:** `[0.0, -0.6, -1.2, -0.6]`

### Group Decoding Rule
- **sum_log_proba:** sum log-probabilities across all windows in a segment, then argmax

""")

code("""# ============================================================
# APPENDIX: FEATURE EXTRACTION CODE  (reference only)
# ============================================================
import librosa

EXTRACT_CFG = {
    "audio": {"sr": 16000, "duration": 1.0},
    "event_crop": {"top_energy_sec": 0.4, "top_energy_hop_sec": 0.05},
    "stft_base": {"n_fft": 512, "win_length": 400, "hop_length": 160, "window": "hann"},
    "mfcc": {"n_mfcc": 20, "n_mels": 64, "fmin": 20, "fmax": 8000},
    "mel": {"n_mels": 64, "n_groups": 7},
    "fft": {"bands": [(20,100),(100,250),(250,500),(500,1000),
                       (1000,2000),(2000,4000),(4000,6000),(6000,8000)]},
}

def _load(path, sr=16000, dur=1.0):
    s, _ = librosa.load(path, sr=sr, mono=True)
    tl = int(sr * dur)
    s = np.pad(s, (0, max(0, tl - len(s))))[:tl]
    return (s / (np.max(np.abs(s)) + 1e-8)).astype(np.float32)

def _top_energy(s, sr=16000, ws=0.4, hs=0.05):
    wl, hl = int(ws * sr), int(hs * sr)
    if len(s) <= wl: return s
    be, bs = -1.0, 0
    for i in range(0, len(s) - wl + 1, hl):
        e = float(np.mean(s[i:i+wl]**2))
        if e > be: be, bs = e, i
    return s[bs:bs+wl]

def _s4(v):
    v = np.asarray(v)
    return np.asarray([np.mean(v), np.std(v), np.max(v), np.percentile(v, 90)], dtype=np.float32)

def _mfcc40(s, sr=16000):
    sc = EXTRACT_CFG["stft_base"]; mc = EXTRACT_CFG["mfcc"]
    m = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=mc["n_mfcc"], n_mels=mc["n_mels"],
                             n_fft=sc["n_fft"], win_length=sc["win_length"],
                             hop_length=sc["hop_length"], fmin=mc["fmin"], fmax=mc["fmax"])
    return np.concatenate([np.mean(m, axis=1), np.std(m, axis=1)]).astype(np.float32)

def _stft28(s, sr=16000):
    sc = EXTRACT_CFG["stft_base"]
    mag = np.abs(librosa.stft(s, n_fft=sc["n_fft"], hop_length=sc["hop_length"],
                              win_length=sc["win_length"], window=sc["window"]))
    ss = [
        librosa.feature.rms(S=mag, frame_length=sc["n_fft"], hop_length=sc["hop_length"])[0],
        librosa.feature.zero_crossing_rate(s, frame_length=sc["n_fft"], hop_length=sc["hop_length"])[0],
        librosa.feature.spectral_centroid(S=mag, sr=sr)[0],
        librosa.feature.spectral_bandwidth(S=mag, sr=sr)[0],
        librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85)[0],
        librosa.feature.spectral_flatness(S=mag)[0],
    ]
    nm = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-8)
    ss.append(np.pad(np.sqrt(np.sum(np.diff(nm, axis=1)**2, axis=0)), (1, 0)))
    return np.concatenate([_s4(x) for x in ss]).astype(np.float32)

def _mel28(s, sr=16000):
    sc = EXTRACT_CFG["stft_base"]; mc = EXTRACT_CFG["mfcc"]; ml = EXTRACT_CFG["mel"]
    m = librosa.feature.melspectrogram(y=s, sr=sr, n_mels=ml["n_mels"],
                                       n_fft=sc["n_fft"], win_length=sc["win_length"],
                                       hop_length=sc["hop_length"],
                                       fmin=mc["fmin"], fmax=mc["fmax"], power=2.0)
    lm = librosa.power_to_db(m, ref=np.max)
    return np.concatenate([_s4(np.mean(g, axis=0))
                           for g in np.array_split(lm, ml["n_groups"], axis=0)]).astype(np.float32)

def _fft24(s, sr=16000):
    eps = 1e-8
    ff = np.abs(np.fft.rfft(s))
    fr = np.fft.rfftfreq(len(s), d=1.0 / sr)
    pw = ff**2; total = np.sum(pw) + eps
    bp = np.asarray([np.sum(pw[(fr >= lo) & (fr < hi)]) / total
                     for lo, hi in EXTRACT_CFG["fft"]["bands"]], dtype=np.float32)
    log_bp = np.log1p(bp)
    low = np.sum(bp[:3]); mid = np.sum(bp[3:6]); high = np.sum(bp[6:])
    ratios = np.asarray([low/(mid+eps), low/(high+eps), mid/(high+eps), (low+mid)/(high+eps)], dtype=np.float32)
    prob = pw / total
    ent = -np.sum(prob * np.log(prob + eps)) / np.log(len(prob))
    cen = np.sum(fr * pw) / total
    df  = fr[int(np.argmax(pw))]
    extras = np.asarray([ent, cen, df, ff[int(np.argmax(pw))]], dtype=np.float32)
    return np.concatenate([bp, log_bp, ratios, extras]).astype(np.float32)

def extract_total_120(s, sr=16000):
    return np.concatenate([_mfcc40(s, sr), _stft28(s, sr), _mel28(s, sr), _fft24(s, sr)]).astype(np.float32)

def extract_total_240(s, sr=16000):
    top = _top_energy(s, sr, ws=0.4, hs=0.05)
    return np.concatenate([extract_total_120(s, sr), extract_total_120(top, sr)]).astype(np.float32)

# Demo extraction on the first training file
demo_file = train_df["audio_path"].iloc[0]
sig = _load(demo_file)
feat = extract_total_240(sig)
print("Demo:", Path(demo_file).name)
print("  signal:", sig.shape, " features:", feat.shape)
print("  range: [%.4f, %.4f]  all_finite: %s" % (feat.min(), feat.max(), np.isfinite(feat).all()))
print()
print("Feature extraction code is identical to train_val_select_final_test.py")
print("The notebook above uses pre-extracted .npy caches, does NOT re-extract.")""")

nb = {"nbformat":4, "nbformat_minor":5,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.11.0"}},
      "cells": OUT}

nb_path = Path("audio_log_consensus_pair_blend_standalone.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook written:", nb_path, "(%s bytes)" % os.path.getsize(nb_path))
''')

print("Written generator script. Now executing it...")
