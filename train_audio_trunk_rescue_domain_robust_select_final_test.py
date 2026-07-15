from __future__ import annotations

"""Audio-only trunk-rescue with domain-robust augmentation and holdout selection.

Combines three clean (non-leaky) upgrade directions on top of the locked
0.7027 anchor pipeline (`audio_lift_source_blend`):

1. Heavy robot-like waveform augmentation views used only at train time:
   - `robot_heavy`: synthetic impulse-response reverb + strong bandlimit +
     motor-hum/broadband noise + tanh drive + time roll.
   - `pitch_speed`: speed/pitch perturbation + bandlimit + noise.
2. Robot-like selection: candidates are scored on a hand/default
   domain-holdout validation split, using the WORST macro-F1 across stress
   views instead of mean OOF, with a leaf-F1 guard.
3. Trunk-focused rescue: a trunk-vs-rest binary detector trained with heavy
   augmentation. Windows the anchor predicts as ambient/twig are flipped to
   trunk when the segment-mean trunk probability is above a locked threshold.

Protocol guarantees:
- Selection uses only hand/default train data (labels, OOF probabilities,
  domain-holdout split). The selection lock JSON is written BEFORE any
  robot/test file is loaded.
- No image/multimodal features. No class words parsed from filenames for
  prediction. Group/specimen keys are used only for grouping, as in the
  locked anchor pipeline.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from tqdm.auto import tqdm

import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_domain_holdout_select_final_test as domain
import train_audio_group_consistency_pair_blend_select_final_test as group_impl
import train_audio_specimen_contact_consensus_select_final_test as spec
import train_audio_specimen_contact_lift_select_final_test as lift
import train_cv_select_final_test as cv
import train_stress_cv_select_final_test as stress
import train_val_select_final_test as base


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)
CONTACT_LABELS = np.asarray([1, 2, 3], dtype=np.int64)
TRUNK_ID = 2
LEAF_ID = 1

HEAVY_VIEWS = ("robot_heavy", "pitch_speed")
EVAL_VIEWS = ("clean", "robot_mix", "bandlimit", "robot_heavy")
RESCUE_THRESHOLDS = (0.55, 0.65, 0.75, 0.85)
RESCUE_TARGETS: dict[str, tuple[int, ...]] = {
    "ambient": (0,),
    "twig": (3,),
    "ambient_twig": (0, 3),
}

# Locked anchor recipe from checkpoints/audio_only_paper_safe_current_0702672_20260706.
ANCHOR_SOURCE_NAME = "report_gate_onehot"
ANCHOR_SOURCE_WEIGHT = 0.05
ANCHOR_BLEND_MODE = "segment_lift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio-only trunk rescue on the locked 0.7027 anchor. Detector is "
            "trained with heavy robot-like augmentation; the rescue rule is "
            "selected on a hand/default domain-holdout split by worst-view "
            "macro-F1 with a leaf guard. Robot/test loads only after the lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="audio_trunk_rescue_domain_robust_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--leaf-val-frac", type=float, default=0.28)
    parser.add_argument("--letwig-val-frac", type=float, default=0.18)
    parser.add_argument("--guard-leaf-drop", type=float, default=0.01)
    parser.add_argument("--selection-margin", type=float, default=0.0)
    parser.add_argument(
        "--clean-feature-cache-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_trainval_select/features"),
    )
    parser.add_argument(
        "--stress-feature-dir",
        type=Path,
        default=Path("outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features"),
    )
    parser.add_argument(
        "--pairwise-oof-cache",
        type=Path,
        default=Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        ),
    )
    parser.add_argument("--force-rebuild-heavy", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def normalize(proba: np.ndarray) -> np.ndarray:
    return spec.normalize(proba)


# ---------------------------------------------------------------------------
# Direction 2: heavy robot-like waveform augmentation (train-time only)
# ---------------------------------------------------------------------------

def synth_impulse_response(rng: np.random.Generator, sr: int) -> np.ndarray:
    duration = float(rng.uniform(0.03, 0.08))
    n = max(8, int(sr * duration))
    tau = float(rng.uniform(0.006, 0.028))
    t = np.arange(n) / sr
    ir = rng.normal(0.0, 1.0, n) * np.exp(-t / tau)
    ir[0] = 1.0
    ir = ir / (np.max(np.abs(ir)) + 1e-8)
    return ir.astype(np.float32)


def motor_hum_noise(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    f0 = float(rng.uniform(45.0, 120.0))
    t = np.arange(n) / sr
    hum = np.zeros(n, dtype=np.float64)
    for k in range(1, 6):
        hum += (1.0 / k) * np.sin(2.0 * np.pi * f0 * k * t + float(rng.uniform(0.0, 2.0 * np.pi)))
    broadband = rng.normal(0.0, 1.0, n)
    mix = float(rng.uniform(0.3, 0.7))
    hum = hum / (np.max(np.abs(hum)) + 1e-8)
    broadband = broadband / (np.max(np.abs(broadband)) + 1e-8)
    return (mix * hum + (1.0 - mix) * broadband).astype(np.float32)


def add_structured_noise_at_snr(
    signal: np.ndarray,
    rng: np.random.Generator,
    snr_db: float,
    sr: int,
) -> np.ndarray:
    noise = motor_hum_noise(rng, len(signal), sr)
    signal_rms = float(np.sqrt(np.mean(signal**2))) + 1e-8
    noise_rms = float(np.sqrt(np.mean(noise**2))) + 1e-8
    target_rms = signal_rms / (10 ** (snr_db / 20.0))
    return (signal + noise * (target_rms / noise_rms)).astype(np.float32)


def speed_perturb(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rate = float(rng.uniform(0.85, 1.15))
    n = len(signal)
    positions = np.arange(n, dtype=np.float64) * rate
    stretched = np.interp(positions, np.arange(n, dtype=np.float64), signal, left=0.0, right=0.0)
    return stretched.astype(np.float32)


def apply_heavy_view(signal: np.ndarray, view: str, key: str, sr: int) -> np.ndarray:
    rng = np.random.default_rng(stress.stable_seed(f"{view}|{key}"))
    output = signal.astype(np.float32).copy()

    if view == "robot_heavy":
        ir = synth_impulse_response(rng, sr)
        output = np.convolve(output, ir)[: len(signal)].astype(np.float32)
        high_hz = float(rng.uniform(3800.0, 5200.0))
        output = stress.fft_bandlimit(output, sr, low_hz=60.0, high_hz=high_hz)
        output = output * float(rng.uniform(0.5, 1.4))
        output = add_structured_noise_at_snr(output, rng, snr_db=float(rng.uniform(10.0, 22.0)), sr=sr)
        drive = float(rng.uniform(1.2, 2.0))
        output = np.tanh(drive * output) / np.tanh(drive)
        shift = int(rng.integers(-2000, 2001))
        output = np.roll(output, shift)
        return stress.normalize_peak(output)

    if view == "pitch_speed":
        output = speed_perturb(output, rng)
        output = stress.fft_bandlimit(output, sr, low_hz=100.0, high_hz=5000.0)
        output = output * float(rng.uniform(0.7, 1.3))
        output = stress.add_noise_at_snr(output, rng, snr_db=float(rng.uniform(18.0, 30.0)))
        return stress.normalize_peak(output)

    raise ValueError(f"Unknown heavy view: {view}")


def heavy_cache_paths(heavy_feature_dir: Path, view: str) -> dict[str, Path]:
    view_dir = heavy_feature_dir / view
    view_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X": view_dir / "X.npy",
        "y": view_dir / "y.npy",
        "paths": view_dir / "paths.npy",
        "metadata": view_dir / "metadata.json",
    }


def build_or_load_heavy_cache(
    frame: pd.DataFrame,
    view: str,
    heavy_feature_dir: Path,
    force_rebuild: bool,
) -> dict[str, np.ndarray]:
    paths = heavy_cache_paths(heavy_feature_dir, view)
    signature = base.manifest_signature(frame)
    ready = all(path.exists() for path in paths.values())
    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        valid = (
            metadata.get("feature_set") == base.FEATURE_SET
            and int(metadata.get("feature_dim", -1)) == base.EXPECTED_DIM
            and int(metadata.get("n_samples", -1)) == len(frame)
            and metadata.get("manifest_signature") == signature
            and metadata.get("heavy_view") == view
        )
        if valid:
            payload = {
                "X": np.load(paths["X"]),
                "y": np.load(paths["y"]),
                "paths": np.load(paths["paths"], allow_pickle=True),
            }
            print(f"Loaded heavy cache {view}: {payload['X'].shape}")
            return payload
        print(f"Heavy cache metadata mismatch for {view}; rebuilding.")

    sr = int(base.CONFIG["audio"]["sr"])
    rows = []
    labels = []
    audio_paths = []
    start = time.perf_counter()
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"Extract heavy/{view}"):
        signal = base.load_audio(row.audio_path, target_sr=sr, duration=base.CONFIG["audio"]["duration"])
        heavy = apply_heavy_view(signal, view=view, key=str(row.audio_path), sr=sr)
        rows.append(stress.extract_features_from_signal(heavy, sr=sr))
        labels.append(int(row.y))
        audio_paths.append(str(row.audio_path))
    extraction_time = time.perf_counter() - start

    payload = {
        "X": np.stack(rows).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray(audio_paths),
    }
    np.save(paths["X"], payload["X"])
    np.save(paths["y"], payload["y"])
    np.save(paths["paths"], payload["paths"])
    metadata = {
        "feature_set": base.FEATURE_SET,
        "feature_dim": base.EXPECTED_DIM,
        "n_samples": len(frame),
        "manifest_signature": signature,
        "heavy_view": view,
        "extraction_time_sec": extraction_time,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved heavy cache {view}: {payload['X'].shape} in {extraction_time:.2f}s")
    return payload


# ---------------------------------------------------------------------------
# Locked anchor pipeline (identical formula to audio_lift_source_blend)
# ---------------------------------------------------------------------------

def anchor_lift_proba(frame: pd.DataFrame, highsr_proba: np.ndarray, pairwise_proba: np.ndarray) -> np.ndarray:
    window = normalize(0.80 * highsr_proba + 0.20 * pairwise_proba)
    segment, window_to_segment = spec.segment_proba_from_window(frame, window)
    specimen_codes = spec.specimen_codes_for_segments(frame, window_to_segment)
    segment = lift.consensus_and_lift(
        segment,
        specimen_codes,
        consensus_threshold=0.45,
        min_contact_segments=1,
        lift_min_mass=0.35,
        lift_floor=0.58,
        lift_confidence=0.45,
    )
    return normalize(segment[window_to_segment])


def segment_lift_postprocess(frame: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    segment, window_to_segment = spec.segment_proba_from_window(frame, proba)
    specimen_codes = spec.specimen_codes_for_segments(frame, window_to_segment)
    segment = lift.consensus_and_lift(
        segment,
        specimen_codes,
        consensus_threshold=0.45,
        min_contact_segments=1,
        lift_min_mass=0.35,
        lift_floor=0.58,
        lift_confidence=0.45,
    )
    return normalize(segment[window_to_segment])


def locked_anchor_pipeline(
    frame: pd.DataFrame,
    highsr_proba: np.ndarray,
    pairwise_proba: np.ndarray,
    gate_onehot_proba: np.ndarray,
) -> np.ndarray:
    anchor = anchor_lift_proba(frame, highsr_proba, pairwise_proba)
    blended = normalize((1.0 - ANCHOR_SOURCE_WEIGHT) * anchor + ANCHOR_SOURCE_WEIGHT * gate_onehot_proba)
    return segment_lift_postprocess(frame, blended)


# ---------------------------------------------------------------------------
# Direction 4: trunk-vs-rest detector with heavy augmentation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorSpec:
    name: str
    factory_key: str
    train_views: tuple[str, ...]


def make_detector_factories(random_state: int) -> dict[str, callable]:
    return {
        "trunk_hgb": lambda: HistGradientBoostingClassifier(
            max_iter=320,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.02,
            class_weight="balanced",
            random_state=random_state,
        ),
        "trunk_extra": lambda: ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def make_detector_specs() -> dict[str, DetectorSpec]:
    specs = [
        DetectorSpec("trunk_hgb__stress3", "trunk_hgb", ("clean", "robot_mix", "bandlimit")),
        DetectorSpec("trunk_hgb__heavy4", "trunk_hgb", ("clean", "robot_mix", "bandlimit", "robot_heavy")),
        DetectorSpec(
            "trunk_hgb__heavy5",
            "trunk_hgb",
            ("clean", "robot_mix", "bandlimit", "robot_heavy", "pitch_speed"),
        ),
        DetectorSpec(
            "trunk_extra__heavy5",
            "trunk_extra",
            ("clean", "robot_mix", "bandlimit", "robot_heavy", "pitch_speed"),
        ),
    ]
    return {item.name: item for item in specs}


def fit_detector(
    detector: DetectorSpec,
    factories: dict[str, callable],
    X_by_view: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
) -> object:
    X_train = np.vstack([X_by_view[view][train_idx] for view in detector.train_views])
    y_binary = (y[train_idx] == TRUNK_ID).astype(np.int64)
    y_train = np.concatenate([y_binary for _ in detector.train_views])
    model = factories[detector.factory_key]()
    model.fit(X_train, y_train)
    return model


def trunk_probability(model: object, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    positive_column = int(np.where(model.classes_ == 1)[0][0])
    return proba[:, positive_column].astype(np.float64)


def segment_mean_trunk(frame: pd.DataFrame, trunk_window: np.ndarray) -> np.ndarray:
    series = pd.Series(trunk_window, index=np.arange(len(trunk_window)))
    groups = pd.Series(frame["group_key"].to_numpy(), index=np.arange(len(trunk_window)))
    return series.groupby(groups).transform("mean").to_numpy(dtype=np.float64)


def apply_trunk_rescue(
    pred: np.ndarray,
    segment_trunk_proba: np.ndarray,
    rescue_from_ids: tuple[int, ...],
    threshold: float,
) -> np.ndarray:
    output = pred.copy()
    mask = np.isin(output, np.asarray(rescue_from_ids, dtype=np.int64)) & (segment_trunk_proba >= threshold)
    output[mask] = TRUNK_ID
    return output


def leaf_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, pred, labels=LABELS, average=None, zero_division=0)[LEAF_ID]
    )


# ---------------------------------------------------------------------------
# Direction 3: worst-view domain-holdout selection
# ---------------------------------------------------------------------------

def score_candidate_views(
    y_val: np.ndarray,
    baseline_pred_val: np.ndarray,
    seg_trunk_by_view: dict[str, np.ndarray] | None,
    rescue_from_ids: tuple[int, ...] | None,
    threshold: float | None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    macro_values = []
    leaf_values = []
    for view in EVAL_VIEWS:
        if seg_trunk_by_view is None:
            pred = baseline_pred_val
        else:
            pred = apply_trunk_rescue(baseline_pred_val, seg_trunk_by_view[view], rescue_from_ids, threshold)
        macro = float(f1_score(y_val, pred, labels=LABELS, average="macro", zero_division=0))
        contact = float(f1_score(y_val, pred, labels=CONTACT_LABELS, average="macro", zero_division=0))
        scores[f"{view}_macro_f1"] = macro
        scores[f"{view}_contact_macro_f1"] = contact
        scores[f"{view}_leaf_f1"] = leaf_f1(y_val, pred)
        macro_values.append(macro)
        leaf_values.append(scores[f"{view}_leaf_f1"])
    scores["worst_macro_f1"] = float(min(macro_values))
    scores["mean_macro_f1"] = float(np.mean(macro_values))
    scores["worst_leaf_f1"] = float(min(leaf_values))
    return scores


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set("total240")

    root_path = base.resolve_root(args.root)
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    heavy_feature_dir = run_dir / "heavy_features"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, heavy_feature_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH               =", root_path.resolve())
    print("RUN_DIR                 =", run_dir.resolve())
    print("Protocol                = trunk rescue; domain-holdout worst-view selection; test after lock")

    # ---- hand/default train only from here ----
    train_csv = base.require_file(root_path / "audio_visual_dataset_default" / "dataset.csv", "hand/default dataset.csv")
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(cv.specimen_group_key)
    train_df["prefix_alpha"] = train_df["audio_file"].map(domain.prefix_alpha)

    clean_feat, _ = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=args.clean_feature_cache_dir,
        force_rebuild=False,
    )
    y = clean_feat["y"]

    X_by_view: dict[str, np.ndarray] = {"clean": clean_feat["X"]}
    for view in ("robot_mix", "bandlimit"):
        payload, _ = stress.build_or_load_stress_cache(
            train_df,
            view=view,
            stress_feature_dir=args.stress_feature_dir,
            force_rebuild=False,
        )
        if not np.array_equal(y, payload["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")
        X_by_view[view] = payload["X"]
    for view in HEAVY_VIEWS:
        payload = build_or_load_heavy_cache(
            train_df,
            view=view,
            heavy_feature_dir=heavy_feature_dir,
            force_rebuild=args.force_rebuild_heavy,
        )
        if not np.array_equal(y, payload["y"]):
            raise AssertionError(f"Label mismatch between clean and {view}")
        X_by_view[view] = payload["X"]

    # Locked anchor pipeline on train OOF probabilities.
    split_path = (
        args.output
        / "audio_feature_benchmarks"
        / "audio_tta_grid_hgb_select"
        / "splits"
        / "hand_train_full_hgb_tta_grid_folds.csv"
    )
    fold_assignment = pd.read_csv(split_path)["cv_fold"].to_numpy(dtype=np.int64)
    train_sources = broad.load_train_sources(args.output, y, fold_assignment, args.random_state)
    highsr_oof = group_impl.load_highsr_oof(args.output)
    pairwise_oof = spec.load_pairwise_oof(args.pairwise_oof_cache)
    baseline_proba = locked_anchor_pipeline(train_df, highsr_oof, pairwise_oof, train_sources[ANCHOR_SOURCE_NAME])
    baseline_pred = baseline_proba.argmax(axis=1).astype(np.int64)

    # Domain-holdout split (robot-like validation built from hand/default only).
    train_idx, val_idx, split_info = domain.make_domain_holdout_split(
        train_df,
        args.random_state,
        args.leaf_val_frac,
        args.letwig_val_frac,
    )
    split_df = train_df.copy()
    split_df["domain_split"] = "train"
    split_df.loc[val_idx, "domain_split"] = "val"
    split_df.to_csv(split_dir / "hand_train_domain_holdout_split.csv", index=False)
    print("Domain holdout:", json.dumps({k: split_info[k] for k in ["train_samples", "val_samples"]}))

    y_val = y[val_idx]
    val_frame = train_df.iloc[val_idx].reset_index(drop=True)
    baseline_pred_val = baseline_pred[val_idx]
    baseline_scores = score_candidate_views(y_val, baseline_pred_val, None, None, None)
    baseline_row = {
        "candidate": "anchor_baseline",
        "detector": "none",
        "rescue_from": "none",
        "threshold": np.nan,
        "eligible": True,
        **baseline_scores,
    }
    print(
        f"Baseline holdout: worst_macro={baseline_scores['worst_macro_f1']:.4f} "
        f"worst_leaf={baseline_scores['worst_leaf_f1']:.4f}"
    )

    # Train detectors on holdout-train rows; score rescue rules on holdout-val.
    factories = make_detector_factories(args.random_state)
    detector_specs = make_detector_specs()
    leaderboard_rows = [baseline_row]
    for detector in detector_specs.values():
        start = time.perf_counter()
        model = fit_detector(detector, factories, X_by_view, y, train_idx)
        fit_time = time.perf_counter() - start
        seg_trunk_by_view = {}
        for view in EVAL_VIEWS:
            trunk_window = trunk_probability(model, X_by_view[view][val_idx])
            seg_trunk_by_view[view] = segment_mean_trunk(val_frame, trunk_window)
        print(f"\nDetector {detector.name} views={detector.train_views} fit={fit_time:.1f}s")
        for target_name, target_ids in RESCUE_TARGETS.items():
            for threshold in RESCUE_THRESHOLDS:
                scores = score_candidate_views(y_val, baseline_pred_val, seg_trunk_by_view, target_ids, threshold)
                eligible = (
                    scores["worst_leaf_f1"] >= baseline_scores["worst_leaf_f1"] - args.guard_leaf_drop
                    and scores["worst_macro_f1"] >= baseline_scores["worst_macro_f1"] + args.selection_margin
                )
                leaderboard_rows.append(
                    {
                        "candidate": f"{detector.name}__{target_name}__t{threshold:.2f}",
                        "detector": detector.name,
                        "rescue_from": target_name,
                        "threshold": float(threshold),
                        "eligible": bool(eligible),
                        "detector_fit_time_sec": fit_time,
                        **scores,
                    }
                )
                print(
                    f"  {target_name:>12} t={threshold:.2f}: worst_macro={scores['worst_macro_f1']:.4f} "
                    f"mean_macro={scores['mean_macro_f1']:.4f} worst_leaf={scores['worst_leaf_f1']:.4f} "
                    f"eligible={eligible}"
                )

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard = leaderboard.sort_values(
        ["eligible", "worst_macro_f1", "mean_macro_f1", "clean_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_holdout_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    eligible_board = leaderboard[leaderboard["eligible"]]
    selected = eligible_board.iloc[0].to_dict() if len(eligible_board) else baseline_row

    method_card = {
        "protocol": "audio_only_trunk_rescue_domain_robust_no_test_until_lock",
        "anchor_recipe": {
            "source_name": ANCHOR_SOURCE_NAME,
            "source_weight": ANCHOR_SOURCE_WEIGHT,
            "blend_mode": ANCHOR_BLEND_MODE,
        },
        "heavy_views": list(HEAVY_VIEWS),
        "eval_views": list(EVAL_VIEWS),
        "selection_rule": "worst-view macro-F1 on domain holdout, leaf guard, fallback to anchor baseline",
        "guard_leaf_drop": float(args.guard_leaf_drop),
        "selection_margin": float(args.selection_margin),
        "allowed_selection_data": "hand/default labels, locked OOF probabilities, hand/default domain holdout",
        "forbidden_selection_data": "robot/test labels or probabilities, image/multimodal features, filename class words",
        "candidate_count": int(len(leaderboard)),
        "split_info": split_info,
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)

    selection_summary = {
        "selected_without_test": selected,
        "baseline_holdout_scores": baseline_scores,
        "leaderboard_path": str(leaderboard_path.resolve()),
        "method_card": str(method_card_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:")
    print(json.dumps({k: selected.get(k) for k in ["candidate", "detector", "rescue_from", "threshold", "worst_macro_f1", "mean_macro_f1"]}, indent=2, default=float), flush=True)

    # ---- robot/test only from here ----
    test_frame, final_sources = broad.load_final_sources(broad.final_prediction_paths(args.output))
    final_root = args.output / "audio_feature_benchmarks"
    highsr_frame = pd.read_csv(
        final_root / "audio_highsr_temporal_tta_select" / "reports" / "audio_highsr_temporal_tta_select_final_test_predictions.csv"
    )
    pair_frame = pd.read_csv(
        final_root / "audio_pairwise_contact_stress_cv_select" / "reports" / "audio_pairwise_contact_stress_cv_select_final_test_predictions.csv"
    )
    proba_columns = [f"proba_{base.ID2LABEL[index]}" for index in LABELS]
    highsr_final = normalize(highsr_frame[proba_columns].to_numpy(dtype=np.float64))
    pair_final = normalize(pair_frame[proba_columns].to_numpy(dtype=np.float64))
    if not np.array_equal(highsr_frame["audio_file"].astype(str).to_numpy(), test_frame["audio_file"].astype(str).to_numpy()):
        raise AssertionError("High-SR final frame is not aligned with source frame")

    final_baseline_proba = locked_anchor_pipeline(test_frame, highsr_final, pair_final, final_sources[ANCHOR_SOURCE_NAME])
    final_baseline_pred = final_baseline_proba.argmax(axis=1).astype(np.int64)

    selected_candidate = str(selected["candidate"])
    if selected_candidate == "anchor_baseline":
        final_pred = final_baseline_pred
        final_seg_trunk = np.zeros(len(final_baseline_pred), dtype=np.float64)
        final_detector_artifact = None
    else:
        test_csv = base.require_file(root_path / "audio_visual_dataset_robo_default" / "dataset.csv", "robot dataset.csv")
        test_df = base.load_manifest(test_csv, "robot_test")
        if not np.array_equal(test_df["audio_file"].astype(str).to_numpy(), test_frame["audio_file"].astype(str).to_numpy()):
            raise AssertionError("Robot/test manifest is not aligned with final source frame")
        test_feat, _ = base.build_or_load_feature_cache(
            test_df,
            "robot_test",
            feature_dir=args.clean_feature_cache_dir,
            force_rebuild=False,
        )
        detector = detector_specs[str(selected["detector"])]
        full_idx = np.arange(len(y))
        start = time.perf_counter()
        final_detector_artifact = fit_detector(detector, factories, X_by_view, y, full_idx)
        print(f"Refit final detector {detector.name} on full train in {time.perf_counter() - start:.1f}s")
        trunk_window_test = trunk_probability(final_detector_artifact, test_feat["X"])
        final_seg_trunk = segment_mean_trunk(test_frame, trunk_window_test)
        rescue_ids = RESCUE_TARGETS[str(selected["rescue_from"])]
        final_pred = apply_trunk_rescue(final_baseline_pred, final_seg_trunk, rescue_ids, float(selected["threshold"]))

    y_test = test_frame["y"].to_numpy(dtype=np.int64)
    final_row = base.make_report_row(
        model_name="audio_trunk_rescue_domain_robust",
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=0.0,
        predict_time_sec=0.0,
    )
    baseline_macro = float(f1_score(y_test, final_baseline_pred, labels=LABELS, average="macro", zero_division=0))
    final_row.update(
        {
            "selected_by": "hand_default_domain_holdout_worst_view",
            "selected_candidate": selected_candidate,
            "selected_detector": selected.get("detector"),
            "selected_rescue_from": selected.get("rescue_from"),
            "selected_threshold": selected.get("threshold"),
            "selected_holdout_worst_macro_f1": selected.get("worst_macro_f1"),
            "baseline_test_macro_f1": baseline_macro,
            "n_rescued": int((final_pred != final_baseline_pred).sum()),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_frame[[column for column in ["audio_file", "audio_path", "label", "y", "group_key", "source"] if column in test_frame]].copy()
    prediction_frame["baseline_pred_y"] = final_baseline_pred.astype(int)
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    prediction_frame["segment_trunk_proba"] = final_seg_trunk
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_baseline_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(y_test, final_pred, labels=LABELS),
        index=base.CLASS_NAMES,
        columns=base.CLASS_NAMES,
    ).to_csv(confusion_path)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": method_card["protocol"],
            "method_card": method_card,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "final_detector_artifact": final_detector_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": method_card["protocol"],
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "method_card": method_card,
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "artifacts": {
            "method_card": str(method_card_path.resolve()),
            "leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)

    print("\nFinal robot/test result after frozen trunk-rescue selection:")
    print(
        pd.DataFrame([final_row])[
            [
                "model",
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "baseline_test_macro_f1",
                "selected_candidate",
                "n_rescued",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved artifacts:")
    print("Leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
