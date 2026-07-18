from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

import train_val_select_final_test as base

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


LABELS = np.asarray([0, 1, 2, 3], dtype=np.int64)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    kind: str
    direct_factory: Callable[[], object] | None = None
    binary_factory: Callable[[], object] | None = None
    contact_factory: Callable[[], object] | None = None
    members: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean classical-ML selection: use only hand/default grouped CV to "
            "select a model/ensemble and class bias, then evaluate once on robot/test."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--feature-set", choices=sorted(base.FEATURE_SPECS), default="total240")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help=(
            "Feature cache directory. If omitted, the script reuses the cache from "
            "total240_trainval_select when present, otherwise it writes under this run."
        ),
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def specimen_group_key(audio_file: str) -> str:
    stem = Path(audio_file).stem
    return re.sub(r"_segment_.*$", "", stem)


def resolve_feature_cache_dir(run_dir: Path, user_cache_dir: Path | None) -> Path:
    if user_cache_dir is not None:
        return user_cache_dir
    reusable = Path("outputs/audio_feature_benchmarks/total240_trainval_select/features")
    if reusable.exists():
        return reusable
    return run_dir / "features"


def label_counts(y: np.ndarray) -> dict[str, int]:
    return {
        base.ID2LABEL[index]: int((y == index).sum())
        for index in LABELS
    }


def make_cv_splits(
    frame: pd.DataFrame,
    n_folds: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds < 3:
        raise ValueError("--n-folds must be at least 3")
    indices = np.arange(len(frame))
    y = frame["y"].to_numpy()
    groups = frame["specimen_group"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )
    splits = []
    for train_idx, val_idx in splitter.split(indices, y, groups):
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise AssertionError(f"CV group leakage detected: {sorted(overlap)[:5]}")
        if len(set(y[val_idx].tolist())) != len(LABELS):
            raise AssertionError("A CV fold is missing at least one class")
        splits.append((np.asarray(train_idx), np.asarray(val_idx)))
    return splits


def proba_aligned(model: object, X: np.ndarray, labels: np.ndarray = LABELS) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not expose predict_proba")
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    output = np.zeros((len(X), len(labels)), dtype=np.float64)
    for column, class_id in enumerate(classes):
        target = int(np.where(labels == class_id)[0][0])
        output[:, target] = raw[:, column]
    output = np.clip(output, 1e-12, 1.0)
    output = output / output.sum(axis=1, keepdims=True)
    return output


def binary_contact_proba_aligned(model: object, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if 1 not in classes:
        raise AssertionError("Binary contact model did not learn the contact class")
    return raw[:, int(np.where(classes == 1)[0][0])]


def hierarchical_proba(
    binary_model: object,
    contact_model: object,
    X: np.ndarray,
) -> np.ndarray:
    contact_probability = np.clip(binary_contact_proba_aligned(binary_model, X), 1e-12, 1.0)
    contact_class_proba = proba_aligned(contact_model, X, labels=np.asarray([1, 2, 3]))
    output = np.zeros((len(X), 4), dtype=np.float64)
    output[:, 0] = 1.0 - contact_probability
    output[:, 1:4] = contact_probability[:, None] * contact_class_proba
    output = np.clip(output, 1e-12, 1.0)
    return output / output.sum(axis=1, keepdims=True)


def predict_with_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    scores = np.log(np.clip(proba, 1e-12, 1.0)) + bias.reshape(1, -1)
    return scores.argmax(axis=1).astype(np.int64)


def tune_class_bias(
    proba: np.ndarray,
    y_true: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if grid is None:
        grid = np.linspace(-1.2, 1.2, 13)

    best_bias = np.zeros(4, dtype=np.float64)
    best_score = f1_score(y_true, predict_with_bias(proba, best_bias), average="macro")

    # Keep ambient anchored and tune contact-class priors first.
    for contact_biases in itertools.product(grid, repeat=3):
        bias = np.asarray([0.0, *contact_biases], dtype=np.float64)
        score = f1_score(y_true, predict_with_bias(proba, bias), average="macro")
        if score > best_score:
            best_score = score
            best_bias = bias

    # Then allow a small ambient offset around the best contact setting.
    for ambient_bias in np.linspace(-0.8, 0.8, 9):
        bias = best_bias.copy()
        bias[0] = ambient_bias
        score = f1_score(y_true, predict_with_bias(proba, bias), average="macro")
        if score > best_score:
            best_score = score
            best_bias = bias

    return best_bias, float(best_score)


def direct_candidates(random_state: int) -> dict[str, CandidateSpec]:
    class_weights = base.CONFIG["class_weights"]
    candidates = {
        "direct_hgb_default": CandidateSpec(
            name="direct_hgb_default",
            kind="direct",
            direct_factory=lambda: HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                class_weight=class_weights,
                random_state=random_state,
            ),
        ),
        "direct_hgb_regularized": CandidateSpec(
            name="direct_hgb_regularized",
            kind="direct",
            direct_factory=lambda: HistGradientBoostingClassifier(
                max_iter=360,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=0.08,
                class_weight=class_weights,
                random_state=random_state,
            ),
        ),
        "direct_hgb_shallow": CandidateSpec(
            name="direct_hgb_shallow",
            kind="direct",
            direct_factory=lambda: HistGradientBoostingClassifier(
                max_iter=420,
                learning_rate=0.03,
                max_leaf_nodes=9,
                min_samples_leaf=35,
                l2_regularization=0.15,
                class_weight=class_weights,
                random_state=random_state,
            ),
        ),
        "direct_extra_trees": CandidateSpec(
            name="direct_extra_trees",
            kind="direct",
            direct_factory=lambda: ExtraTreesClassifier(
                n_estimators=900,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight=class_weights,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "direct_lightgbm_regularized": CandidateSpec(
            name="direct_lightgbm_regularized",
            kind="direct",
            direct_factory=lambda: LGBMClassifier(
                objective="multiclass",
                num_class=len(base.CLASS_NAMES),
                n_estimators=650,
                learning_rate=0.025,
                num_leaves=15,
                min_child_samples=35,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                class_weight=class_weights,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        "direct_xgboost_regularized": CandidateSpec(
            name="direct_xgboost_regularized",
            kind="direct",
            direct_factory=lambda: XGBClassifier(
                objective="multi:softprob",
                num_class=len(base.CLASS_NAMES),
                n_estimators=500,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=3,
                subsample=0.82,
                colsample_bytree=0.82,
                reg_lambda=2.0,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "direct_rbf_svm": CandidateSpec(
            name="direct_rbf_svm",
            kind="direct",
            direct_factory=lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            C=8,
                            gamma="scale",
                            class_weight=class_weights,
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    }
    return candidates


def hierarchical_candidates(random_state: int) -> dict[str, CandidateSpec]:
    return {
        "hier_hgb_hgb": CandidateSpec(
            name="hier_hgb_hgb",
            kind="hierarchical",
            binary_factory=lambda: HistGradientBoostingClassifier(
                max_iter=240,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=0.05,
                class_weight="balanced",
                random_state=random_state,
            ),
            contact_factory=lambda: HistGradientBoostingClassifier(
                max_iter=380,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=0.1,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "hier_hgb_lightgbm": CandidateSpec(
            name="hier_hgb_lightgbm",
            kind="hierarchical",
            binary_factory=lambda: HistGradientBoostingClassifier(
                max_iter=240,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=0.05,
                class_weight="balanced",
                random_state=random_state,
            ),
            contact_factory=lambda: LGBMClassifier(
                objective="multiclass",
                n_estimators=520,
                learning_rate=0.025,
                num_leaves=15,
                min_child_samples=24,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        "hier_extra_hgb": CandidateSpec(
            name="hier_extra_hgb",
            kind="hierarchical",
            binary_factory=lambda: ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            contact_factory=lambda: HistGradientBoostingClassifier(
                max_iter=380,
                learning_rate=0.035,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=0.1,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "hier_hgb_svm": CandidateSpec(
            name="hier_hgb_svm",
            kind="hierarchical",
            binary_factory=lambda: HistGradientBoostingClassifier(
                max_iter=240,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=0.05,
                class_weight="balanced",
                random_state=random_state,
            ),
            contact_factory=lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            C=6,
                            gamma="scale",
                            class_weight="balanced",
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    }


def make_candidates(random_state: int) -> dict[str, CandidateSpec]:
    candidates = {}
    candidates.update(direct_candidates(random_state))
    candidates.update(hierarchical_candidates(random_state))
    return candidates


def fit_predict_direct_oof(
    spec: CandidateSpec,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    oof_proba = np.zeros((len(y), 4), dtype=np.float64)
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        model = spec.direct_factory()
        model.fit(X[train_idx], y[train_idx])
        train_time = time.perf_counter() - start
        oof_proba[val_idx] = proba_aligned(model, X[val_idx])
        pred = oof_proba[val_idx].argmax(axis=1)
        fold_rows.append(
            {
                "candidate": spec.name,
                "fold": fold_id,
                "macro_f1": f1_score(y[val_idx], pred, average="macro"),
                "train_time_sec": train_time,
                "val_samples": len(val_idx),
            }
        )
        print(
            f"  fold {fold_id}: macro_f1={fold_rows[-1]['macro_f1']:.4f} "
            f"time={train_time:.2f}s"
        )
    return oof_proba, fold_rows


def fit_predict_hierarchical_oof(
    spec: CandidateSpec,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    oof_proba = np.zeros((len(y), 4), dtype=np.float64)
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        binary_model = spec.binary_factory()
        contact_model = spec.contact_factory()

        y_binary = (y[train_idx] > 0).astype(np.int64)
        binary_model.fit(X[train_idx], y_binary)

        contact_train_idx = train_idx[y[train_idx] > 0]
        contact_model.fit(X[contact_train_idx], y[contact_train_idx])

        train_time = time.perf_counter() - start
        oof_proba[val_idx] = hierarchical_proba(binary_model, contact_model, X[val_idx])
        pred = oof_proba[val_idx].argmax(axis=1)
        fold_rows.append(
            {
                "candidate": spec.name,
                "fold": fold_id,
                "macro_f1": f1_score(y[val_idx], pred, average="macro"),
                "train_time_sec": train_time,
                "val_samples": len(val_idx),
            }
        )
        print(
            f"  fold {fold_id}: macro_f1={fold_rows[-1]['macro_f1']:.4f} "
            f"time={train_time:.2f}s"
        )
    return oof_proba, fold_rows


def evaluate_oof_candidate(
    spec: CandidateSpec,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, np.ndarray, list[dict]]:
    print(f"\nOOF candidate: {spec.name}")
    if spec.kind == "direct":
        oof_proba, fold_rows = fit_predict_direct_oof(spec, X, y, splits)
    elif spec.kind == "hierarchical":
        oof_proba, fold_rows = fit_predict_hierarchical_oof(spec, X, y, splits)
    else:
        raise ValueError(f"Unsupported OOF kind: {spec.kind}")

    default_pred = oof_proba.argmax(axis=1)
    default_macro_f1 = f1_score(y, default_pred, average="macro")
    bias, tuned_macro_f1 = tune_class_bias(oof_proba, y)
    tuned_pred = predict_with_bias(oof_proba, bias)
    row = base.make_report_row(
        model_name=spec.name,
        split_name="oof_cv",
        y_true=y,
        y_pred=tuned_pred,
        train_time_sec=sum(item["train_time_sec"] for item in fold_rows),
        predict_time_sec=0.0,
    )
    row.update(
        {
            "candidate_kind": spec.kind,
            "default_oof_macro_f1": default_macro_f1,
            "tuned_oof_macro_f1": tuned_macro_f1,
            "class_bias_json": json.dumps(bias.tolist()),
            "members_json": json.dumps(list(spec.members)),
        }
    )
    print(
        f"  OOF default={default_macro_f1:.4f} | tuned={tuned_macro_f1:.4f} "
        f"| bias={np.round(bias, 3).tolist()}"
    )
    return row, oof_proba, fold_rows


def build_ensemble_rows(
    leaderboard: pd.DataFrame,
    oof_probas: dict[str, np.ndarray],
    y: np.ndarray,
) -> tuple[list[dict], dict[str, CandidateSpec], dict[str, np.ndarray]]:
    rows = []
    specs = {}
    probas = {}
    ranked = leaderboard.sort_values("tuned_oof_macro_f1", ascending=False)["model"].tolist()
    recipes = {
        "ensemble_top3_uniform": tuple(ranked[:3]),
        "ensemble_top5_uniform": tuple(ranked[:5]),
        "ensemble_tree_top4_uniform": tuple(
            name for name in ranked if "svm" not in name and "logistic" not in name
        )[:4],
    }
    for name, members in recipes.items():
        if len(members) < 2:
            continue
        proba = np.mean([oof_probas[member] for member in members], axis=0)
        bias, tuned_macro_f1 = tune_class_bias(proba, y)
        pred = predict_with_bias(proba, bias)
        row = base.make_report_row(
            model_name=name,
            split_name="oof_cv",
            y_true=y,
            y_pred=pred,
            train_time_sec=0.0,
            predict_time_sec=0.0,
        )
        row.update(
            {
                "candidate_kind": "ensemble",
                "default_oof_macro_f1": f1_score(y, proba.argmax(axis=1), average="macro"),
                "tuned_oof_macro_f1": tuned_macro_f1,
                "class_bias_json": json.dumps(bias.tolist()),
                "members_json": json.dumps(list(members)),
            }
        )
        rows.append(row)
        specs[name] = CandidateSpec(name=name, kind="ensemble", members=members)
        probas[name] = proba
        print(
            f"\nOOF ensemble: {name} members={members} "
            f"default={row['default_oof_macro_f1']:.4f} tuned={tuned_macro_f1:.4f}"
        )
    return rows, specs, probas


def fit_predict_full(spec: CandidateSpec, candidates: dict[str, CandidateSpec], X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    if spec.kind == "direct":
        model = spec.direct_factory()
        model.fit(X_train, y_train)
        return proba_aligned(model, X_eval)

    if spec.kind == "hierarchical":
        binary_model = spec.binary_factory()
        binary_model.fit(X_train, (y_train > 0).astype(np.int64))
        contact_idx = np.where(y_train > 0)[0]
        contact_model = spec.contact_factory()
        contact_model.fit(X_train[contact_idx], y_train[contact_idx])
        return hierarchical_proba(binary_model, contact_model, X_eval)

    if spec.kind == "ensemble":
        member_probas = [
            fit_predict_full(candidates[member], candidates, X_train, y_train, X_eval)
            for member in spec.members
        ]
        return np.mean(member_probas, axis=0)

    raise ValueError(f"Unsupported final kind: {spec.kind}")


def fit_final_artifact(spec: CandidateSpec, candidates: dict[str, CandidateSpec], X_train: np.ndarray, y_train: np.ndarray) -> object:
    if spec.kind == "direct":
        model = spec.direct_factory()
        model.fit(X_train, y_train)
        return {"kind": "direct", "model": model}

    if spec.kind == "hierarchical":
        binary_model = spec.binary_factory()
        binary_model.fit(X_train, (y_train > 0).astype(np.int64))
        contact_idx = np.where(y_train > 0)[0]
        contact_model = spec.contact_factory()
        contact_model.fit(X_train[contact_idx], y_train[contact_idx])
        return {
            "kind": "hierarchical",
            "binary_model": binary_model,
            "contact_model": contact_model,
        }

    if spec.kind == "ensemble":
        return {
            "kind": "ensemble",
            "members": {
                member: fit_final_artifact(candidates[member], candidates, X_train, y_train)
                for member in spec.members
            },
        }

    raise ValueError(f"Unsupported artifact kind: {spec.kind}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base.CONFIG["random_state"] = args.random_state
    base.configure_feature_set(args.feature_set)

    root_path = base.resolve_root(args.root)
    run_slug = f"{args.feature_set}_clean_cv_select"
    run_dir = args.output / "audio_feature_benchmarks" / run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    split_dir = run_dir / "splits"
    for directory in [run_dir, report_dir, model_dir, split_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    feature_cache_dir = resolve_feature_cache_dir(run_dir, args.feature_cache_dir)

    print("ROOT_PATH         =", root_path.resolve())
    print("RUN_DIR           =", run_dir.resolve())
    print("FEATURE_CACHE_DIR =", feature_cache_dir.resolve())
    print("Feature set       =", base.FEATURE_SPEC["display_name"])
    print("Selection         = grouped OOF CV macro_f1_4class, then OOF-tuned class bias")

    train_csv = base.require_file(
        root_path / "audio_visual_dataset_default" / "dataset.csv",
        "hand/default dataset.csv",
    )
    train_df = base.load_manifest(train_csv, "hand_train")
    train_df["specimen_group"] = train_df["audio_file"].map(specimen_group_key)

    train_feat, train_feature_timing = base.build_or_load_feature_cache(
        train_df,
        "hand_train_full",
        feature_dir=feature_cache_dir,
        force_rebuild=args.force_rebuild,
    )
    X_train = train_feat["X"]
    y_train = train_feat["y"]

    splits = make_cv_splits(train_df, args.n_folds, args.random_state)
    fold_rows = []
    train_df.assign(
        cv_fold=-1,
    ).to_csv(split_dir / "hand_train_full_with_groups.csv", index=False)
    fold_assignment = np.full(len(train_df), -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(splits, start=1):
        fold_assignment[val_idx] = fold_id
    train_df.assign(cv_fold=fold_assignment).to_csv(
        split_dir / "hand_train_full_cv_folds.csv",
        index=False,
    )

    split_summary = {
        "protocol": "hand/default grouped CV only for selection; robot/test loaded after selection",
        "n_train_samples": len(train_df),
        "n_specimen_groups": int(train_df["specimen_group"].nunique()),
        "n_folds": args.n_folds,
        "label_counts": label_counts(y_train),
        "folds": [
            {
                "fold": fold_id,
                "train_samples": int(len(train_idx)),
                "val_samples": int(len(val_idx)),
                "val_label_counts": label_counts(y_train[val_idx]),
                "train_groups": int(train_df.iloc[train_idx]["specimen_group"].nunique()),
                "val_groups": int(train_df.iloc[val_idx]["specimen_group"].nunique()),
            }
            for fold_id, (train_idx, val_idx) in enumerate(splits, start=1)
        ],
    }
    write_json(report_dir / f"{run_slug}_cv_split_summary.json", split_summary)

    candidates = make_candidates(args.random_state)
    leaderboard_rows = []
    oof_probas = {}
    for spec in candidates.values():
        row, proba, rows = evaluate_oof_candidate(spec, X_train, y_train, splits)
        leaderboard_rows.append(row)
        oof_probas[spec.name] = proba
        fold_rows.extend(rows)

    initial_leaderboard = pd.DataFrame(leaderboard_rows)
    ensemble_rows, ensemble_specs, ensemble_probas = build_ensemble_rows(
        initial_leaderboard,
        oof_probas,
        y_train,
    )
    leaderboard_rows.extend(ensemble_rows)
    candidates.update(ensemble_specs)
    oof_probas.update(ensemble_probas)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["tuned_oof_macro_f1", "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    fold_report = pd.DataFrame(fold_rows)
    leaderboard_path = report_dir / f"{run_slug}_oof_leaderboard.csv"
    fold_report_path = report_dir / f"{run_slug}_fold_report.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    fold_report.to_csv(fold_report_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selected_spec = candidates[selected_name]
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_kind": selected_spec.kind,
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "fold_report_path": str(fold_report_path.resolve()),
        "split_summary_path": str((report_dir / f"{run_slug}_cv_split_summary.json").resolve()),
    }
    selection_path = report_dir / f"{run_slug}_selected_without_test.json"
    write_json(selection_path, selection_summary)
    print("\nSelected using hand/default CV only:")
    print(json.dumps(selection_summary, indent=2, default=float))

    # Robot/test is intentionally loaded only after the selection artifact above is written.
    test_csv = base.require_file(
        root_path / "audio_visual_dataset_robo_default" / "dataset.csv",
        "robot dataset.csv",
    )
    test_df = base.load_manifest(test_csv, "robot_test")
    test_feat, test_feature_timing = base.build_or_load_feature_cache(
        test_df,
        "robot_test",
        feature_dir=feature_cache_dir,
        force_rebuild=args.force_rebuild,
    )
    X_test = test_feat["X"]
    y_test = test_feat["y"]

    start = time.perf_counter()
    final_proba = fit_predict_full(selected_spec, candidates, X_train, y_train, X_test)
    final_train_predict_time = time.perf_counter() - start
    final_pred = predict_with_bias(final_proba, selected_bias)
    final_row = base.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=y_test,
        y_pred=final_pred,
        train_time_sec=final_train_predict_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_default_grouped_oof_cv_only",
            "selected_oof_macro_f1": selected["tuned_oof_macro_f1"],
            "selected_bias_json": json.dumps(selected_bias.tolist()),
            "feature_cache_dir": str(feature_cache_dir.resolve()),
        }
    )

    final_report_path = report_dir / f"{run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)

    predictions_path = report_dir / f"{run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "audio_path", "label", "y", "group_key", "source"]
    ].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(base.ID2LABEL)
    for class_id, class_name in base.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    final_artifact = fit_final_artifact(selected_spec, candidates, X_train, y_train)
    bundle_path = model_dir / f"{run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "config": base.CONFIG,
            "feature_set": base.FEATURE_SET,
            "feature_spec": base.FEATURE_SPEC,
            "feature_names": base.FEATURE_NAMES,
            "label_map": base.LABEL_MAP,
            "id2label": base.ID2LABEL,
            "selection_summary": selection_summary,
            "final_test_report": final_row,
            "selected_artifact": final_artifact,
        },
        bundle_path,
    )

    protocol_summary = {
        "protocol": "clean_cv_select_no_test_until_final",
        "root_path": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "feature_cache_dir": str(feature_cache_dir.resolve()),
        "train_feature_timing": train_feature_timing,
        "test_feature_timing": test_feature_timing,
        "selection_summary": selection_summary,
        "final_test_report": final_row,
        "artifacts": {
            "oof_leaderboard": str(leaderboard_path.resolve()),
            "fold_report": str(fold_report_path.resolve()),
            "selected_without_test": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
        },
    }
    summary_path = report_dir / f"{run_slug}_protocol_summary.json"
    write_json(summary_path, protocol_summary)

    print("\nFinal robot/test result after frozen CV selection:")
    print(pd.DataFrame([final_row])[
        [
            "model",
            "accuracy_4class",
            "macro_f1_4class",
            "binary_accuracy",
            "binary_macro_f1",
            "selected_oof_macro_f1",
        ]
    ].to_string(index=False))
    print("\nSaved artifacts:")
    print("OOF leaderboard:", leaderboard_path.resolve())
    print("Selection lock:", selection_path.resolve())
    print("Final test report:", final_report_path.resolve())
    print("Bundle:", bundle_path.resolve())


if __name__ == "__main__":
    main()
