from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

import train_image_group_majority_select_final_test as gm
import train_image_handcrafted_ml_select_final_test as img
import train_image_timm_transfer_select_final_test as timm_transfer


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    view: str
    label_policy: str
    factory: Callable[[], object]


@dataclass
class ExactImageSplit:
    X_views: dict[str, np.ndarray]
    y_group_by_policy: dict[str, np.ndarray]
    group_hashes: np.ndarray
    group_sizes: np.ndarray
    group_counts: np.ndarray
    row_y: np.ndarray
    row_to_group: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only timm embedding model with exact-duplicate image denoising. "
            "Rows with identical image bytes are collapsed inside hand/default train/val, "
            "candidate label policy/head/bias are selected on validation only, a lock is "
            "written, and robot/test is loaded only after that lock."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_timm_dedup_group_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["specimen", "segment"], default="specimen")
    parser.add_argument(
        "--backbones",
        default=(
            "vit_base_patch14_dinov2.lvd142m,"
            "vit_base_patch16_clip_224.openai_ft_in1k,"
            "eva02_base_patch14_224.mim_in22k,"
            "convnext_base.fb_in22k_ft_in1k"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("outputs/image_timm_features"))
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["regularized_val_score", "macro_f1_4class", "contact_macro_f1", "binary_macro_f1"],
        default="regularized_val_score",
    )
    parser.add_argument("--bias-penalty", type=float, default=0.025)
    parser.add_argument("--max-combo-views", type=int, default=3)
    return parser.parse_args()


def label_from_counts(counts: np.ndarray, policy: str) -> int:
    counts = counts.astype(np.int64)
    if policy == "majority":
        return int(counts.argmax())
    contact_counts = counts[img.CONTACT_LABELS]
    contact_total = int(contact_counts.sum())
    if policy == "contact_if_any":
        if contact_total > 0:
            return int(img.CONTACT_LABELS[int(contact_counts.argmax())])
        return 0
    if policy == "contact_if_ge_ambient":
        if contact_total >= int(counts[0]) and contact_total > 0:
            return int(img.CONTACT_LABELS[int(contact_counts.argmax())])
        return 0
    if policy == "contact_if_gt_ambient":
        if contact_total > int(counts[0]) and contact_total > 0:
            return int(img.CONTACT_LABELS[int(contact_counts.argmax())])
        return 0
    if policy == "ambient_if_tie_else_majority":
        max_count = int(counts.max())
        winners = np.flatnonzero(counts == max_count)
        return 0 if 0 in winners else int(winners[0])
    raise ValueError(f"Unknown label policy: {policy}")


def build_exact_image_split(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    row_views: dict[str, np.ndarray],
    label_policies: list[str],
) -> ExactImageSplit:
    sub = frame.iloc[row_indices].copy()
    sub["row_position"] = row_indices
    group_items = list(sub.groupby("image_hash", sort=False))
    X_views = {name: [] for name in row_views}
    y_by_policy = {policy: [] for policy in label_policies}
    group_hashes = []
    group_sizes = []
    group_counts = []
    row_to_group_lookup: dict[int, int] = {}

    for group_index, (hash_value, rows) in enumerate(group_items):
        positions = rows["row_position"].to_numpy(dtype=np.int64)
        counts = np.bincount(rows["y"].to_numpy(dtype=np.int64), minlength=len(img.CLASS_NAMES))
        for name, matrix in row_views.items():
            X_views[name].append(matrix[positions].mean(axis=0))
        for policy in label_policies:
            y_by_policy[policy].append(label_from_counts(counts, policy))
        group_hashes.append(str(hash_value))
        group_sizes.append(int(len(rows)))
        group_counts.append(counts)
        for position in positions:
            row_to_group_lookup[int(position)] = group_index

    return ExactImageSplit(
        X_views={name: np.vstack(values).astype(np.float32) for name, values in X_views.items()},
        y_group_by_policy={policy: np.asarray(values, dtype=np.int64) for policy, values in y_by_policy.items()},
        group_hashes=np.asarray(group_hashes),
        group_sizes=np.asarray(group_sizes, dtype=np.int64),
        group_counts=np.vstack(group_counts).astype(np.int64),
        row_y=sub["y"].to_numpy(dtype=np.int64),
        row_to_group=np.asarray([row_to_group_lookup[int(position)] for position in row_indices], dtype=np.int64),
    )


def add_combo_views(row_views: dict[str, np.ndarray], max_combo_views: int) -> dict[str, np.ndarray]:
    output = dict(row_views)
    names = list(row_views)
    if len(names) >= 2:
        output["combo_all"] = np.concatenate([row_views[name] for name in names], axis=1)
    if len(names) >= 2 and max_combo_views >= 2:
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                key = f"combo__{timm_transfer.safe_name(first)}__{timm_transfer.safe_name(second)}"
                output[key] = np.concatenate([row_views[first], row_views[second]], axis=1)
    return output


def make_candidates(
    views: dict[str, np.ndarray],
    label_policies: list[str],
    random_state: int,
) -> dict[str, CandidateSpec]:
    candidates: dict[str, CandidateSpec] = {}

    def add(name: str, view: str, policy: str, factory: Callable[[], object]) -> None:
        candidates[name] = CandidateSpec(name=name, view=view, label_policy=policy, factory=factory)

    def logreg(c: float) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=6000,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def knn(k: int, metric: str) -> Pipeline:
        steps = []
        if metric == "cosine":
            steps.append(("norm", Normalizer()))
        else:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", KNeighborsClassifier(n_neighbors=k, weights="distance", metric=metric)))
        return Pipeline(steps)

    def rbf_svc(c: float) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=c,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    for view in views:
        for policy in label_policies:
            safe_view = timm_transfer.safe_name(view)
            prefix = f"{safe_view}__{policy}"
            add(f"{prefix}__logreg_C0.1", view, policy, lambda: logreg(0.1))
            add(f"{prefix}__logreg_C1", view, policy, lambda: logreg(1.0))
            add(f"{prefix}__knn1_cosine", view, policy, lambda: knn(1, "cosine"))
            add(f"{prefix}__knn3_cosine", view, policy, lambda: knn(3, "cosine"))
            add(f"{prefix}__knn5_cosine", view, policy, lambda: knn(5, "cosine"))
            add(f"{prefix}__knn3_euclidean", view, policy, lambda: knn(3, "euclidean"))
            add(f"{prefix}__rbf_svc_C1", view, policy, lambda: rbf_svc(1.0))
            add(
                f"{prefix}__extra_trees",
                view,
                policy,
                lambda: ExtraTreesClassifier(
                    n_estimators=900,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
            add(
                f"{prefix}__random_forest",
                view,
                policy,
                lambda: RandomForestClassifier(
                    n_estimators=700,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
    return candidates


def evaluate_candidate(
    spec: CandidateSpec,
    train_split: ExactImageSplit,
    val_split: ExactImageSplit,
    bias_penalty: float,
) -> list[dict[str, object]]:
    model = spec.factory()
    y_train = train_split.y_group_by_policy[spec.label_policy]
    start = time.perf_counter()
    model.fit(train_split.X_views[spec.view], y_train)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    group_proba = img.proba_aligned(model, val_split.X_views[spec.view])
    row_proba = group_proba[val_split.row_to_group]
    predict_time = time.perf_counter() - start
    tuned_bias, _ = img.tune_class_bias(row_proba, val_split.row_y)
    rows: list[dict[str, object]] = []
    for bias_mode, bias in [
        ("none", np.zeros(len(img.CLASS_NAMES), dtype=np.float64)),
        ("tuned", tuned_bias),
    ]:
        pred = img.predict_with_bias(row_proba, bias)
        row = img.make_report_row(
            model_name=spec.name,
            split_name="hand_val",
            y_true=val_split.row_y,
            y_pred=pred,
            feature_set=spec.view,
            feature_name=f"{spec.view}_exact_image_dedup",
            n_features=train_split.X_views[spec.view].shape[1],
            train_time_sec=train_time,
            predict_time_sec=predict_time,
        )
        bias_l1 = float(np.abs(bias).sum())
        row.update(
            {
                "view": spec.view,
                "label_policy": spec.label_policy,
                "bias_mode": bias_mode,
                "class_bias_json": json.dumps(bias.tolist()),
                "bias_l1": bias_l1,
                "regularized_val_score": float(row["macro_f1_4class"] - bias_penalty * bias_l1),
                "train_exact_image_groups": int(len(train_split.group_hashes)),
                "val_exact_image_groups": int(len(val_split.group_hashes)),
            }
        )
        rows.append(row)
    return rows


def fit_predict_final(
    spec: CandidateSpec,
    train_full_split: ExactImageSplit,
    test_split: ExactImageSplit,
    selected_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, object, float, float]:
    model = clone(spec.factory())
    y_train = train_full_split.y_group_by_policy[spec.label_policy]
    start = time.perf_counter()
    model.fit(train_full_split.X_views[spec.view], y_train)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    group_proba = img.proba_aligned(model, test_split.X_views[spec.view])
    row_proba = group_proba[test_split.row_to_group]
    row_pred = img.predict_with_bias(row_proba, selected_bias)
    predict_time = time.perf_counter() - start
    return row_pred, row_proba, model, train_time, predict_time


def write_exact_group_manifest(path: Path, frame: pd.DataFrame, split: ExactImageSplit, policy: str) -> None:
    rows = []
    by_hash = frame.groupby("image_hash", sort=False)
    y_policy = split.y_group_by_policy.get(policy)
    for group_index, hash_value in enumerate(split.group_hashes):
        group = by_hash.get_group(hash_value)
        counts = split.group_counts[group_index]
        majority_y = int(counts.argmax())
        assigned_y = int(y_policy[group_index]) if y_policy is not None else majority_y
        rows.append(
            {
                "image_hash": hash_value,
                "representative_image_file": group.iloc[0]["image_file"],
                "representative_image_path": str(group.iloc[0]["image_path"]),
                "majority_label": img.ID2LABEL[majority_y],
                "assigned_label": img.ID2LABEL[assigned_y],
                "n_rows": int(split.group_sizes[group_index]),
                "label_counts_json": json.dumps({img.ID2LABEL[i]: int(counts[i]) for i in range(len(img.CLASS_NAMES))}),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    timm_transfer.set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    group_dir = run_dir / "groups"
    for directory in [run_dir, report_dir, model_dir, group_dir, args.feature_cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve(), flush=True)
    print("RUN_DIR   =", run_dir.resolve(), flush=True)
    print("DEVICE    =", device, flush=True)
    print("Protocol  = image-only timm exact-image dedup selection; robot/test after lock", flush=True)

    train_df = img.load_image_manifest(
        img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv"),
        train_dataset_dir,
        "hand_train",
    )
    train_df = gm.add_image_hashes(train_df)
    train_idx, val_idx, split_info = img.split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = img.save_split_manifests(run_dir, train_df, None, train_idx, val_idx)

    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    row_views: dict[str, np.ndarray] = {}
    train_feature_timing: dict[str, dict[str, object]] = {}
    for backbone in backbones:
        payload, timing = timm_transfer.extract_timm_features(
            train_df,
            "hand_train_full",
            backbone,
            args.feature_cache_dir,
            args.batch_size,
            args.num_workers,
            args.force_rebuild_features,
            device,
        )
        if not np.array_equal(payload["y"], train_df["y"].to_numpy()):
            raise AssertionError(f"{backbone}: train feature labels do not align")
        row_views[backbone] = payload["X"]
        train_feature_timing[backbone] = timing
    row_views = add_combo_views(row_views, args.max_combo_views)

    label_policies = [
        "majority",
        "contact_if_gt_ambient",
        "contact_if_ge_ambient",
        "contact_if_any",
        "ambient_if_tie_else_majority",
    ]
    train_split = build_exact_image_split(train_df, train_idx, row_views, label_policies)
    val_split = build_exact_image_split(train_df, val_idx, row_views, label_policies)
    train_full_split = build_exact_image_split(train_df, np.arange(len(train_df)), row_views, label_policies)

    split_summary = {
        "protocol": "image_only_timm_exact_image_dedup_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "train_full_rows": int(len(train_df)),
        "train_full_duplicate_summary": gm.duplicate_summary(train_df),
        "train_exact_image_groups": int(len(train_split.group_hashes)),
        "val_exact_image_groups": int(len(val_split.group_hashes)),
        "train_full_exact_image_groups": int(len(train_full_split.group_hashes)),
        "views": {name: int(matrix.shape[1]) for name, matrix in row_views.items()},
        "label_policies": label_policies,
        "train_feature_timing": train_feature_timing,
        "bias_penalty": args.bias_penalty,
        "split_paths": split_paths,
        "forbidden_inputs": [
            "audio features",
            "multimodal features",
            "filename label tokens as predictive features",
            "robot/test labels before selection lock",
        ],
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)

    candidates = make_candidates(train_split.X_views, label_policies, args.random_state)
    leaderboard_rows: list[dict[str, object]] = []
    for spec in candidates.values():
        print(f"\nVAL candidate: {spec.name}", flush=True)
        try:
            rows = evaluate_candidate(spec, train_split, val_split, args.bias_penalty)
        except Exception as exc:
            print(f"  failed: {exc}", flush=True)
            continue
        for row in rows:
            print(
                f"  {row['bias_mode']:5s} macro={row['macro_f1_4class']:.4f} "
                f"reg={row['regularized_val_score']:.4f} acc={row['accuracy_4class']:.4f}",
                flush=True,
            )
        leaderboard_rows.extend(rows)
    if not leaderboard_rows:
        raise RuntimeError("No candidates completed")

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        [args.selection_metric, "macro_f1_4class", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_val_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = candidates[selected_name]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    write_exact_group_manifest(
        group_dir / f"{args.run_slug}_train_full_groups_selected_policy.csv",
        train_df,
        train_full_split,
        selected_spec.label_policy,
    )
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_view": selected_spec.view,
        "selected_label_policy": selected_spec.label_policy,
        "selected_bias_mode": selected["bias_mode"],
        "selected_bias": selected_bias.tolist(),
        "selection_metric": args.selection_metric,
        "leaderboard_path": str(leaderboard_path.resolve()),
        "split_summary_path": str(split_summary_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    img.write_json(selection_path, selection_summary)
    print("\nSelection lock written before loading robot/test:", flush=True)
    print(json.dumps(selection_summary, indent=2, default=float), flush=True)

    test_df = img.load_image_manifest(
        img.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv"),
        test_dataset_dir,
        "robot_test",
    )
    test_df = gm.add_image_hashes(test_df)
    split_paths = img.save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)
    needed_backbones = [backbone for backbone in backbones if backbone in selected_spec.view or selected_spec.view == "combo_all"]
    if selected_spec.view.startswith("combo__"):
        needed_backbones = [name for name in backbones if name in selected_spec.view]
    test_row_views: dict[str, np.ndarray] = {}
    test_feature_timing: dict[str, dict[str, object]] = {}
    for backbone in needed_backbones:
        payload, timing = timm_transfer.extract_timm_features(
            test_df,
            "robot_test",
            backbone,
            args.feature_cache_dir,
            args.batch_size,
            args.num_workers,
            args.force_rebuild_features,
            device,
        )
        if not np.array_equal(payload["y"], test_df["y"].to_numpy()):
            raise AssertionError(f"{backbone}: test feature labels do not align")
        test_row_views[backbone] = payload["X"]
        test_feature_timing[backbone] = timing
    test_row_views = add_combo_views(test_row_views, args.max_combo_views)
    test_split = build_exact_image_split(
        test_df,
        np.arange(len(test_df)),
        {selected_spec.view: test_row_views[selected_spec.view]},
        label_policies,
    )
    write_exact_group_manifest(
        group_dir / f"{args.run_slug}_robot_test_groups_for_audit.csv",
        test_df,
        test_split,
        selected_spec.label_policy,
    )

    final_pred, final_proba, final_model, final_fit_time, final_predict_time = fit_predict_final(
        selected_spec,
        train_full_split,
        test_split,
        selected_bias,
    )
    final_row = img.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_split.row_y,
        y_pred=final_pred,
        feature_set=selected_spec.view,
        feature_name=f"{selected_spec.view}_exact_image_dedup",
        n_features=train_full_split.X_views[selected_spec.view].shape[1],
        train_time_sec=final_fit_time,
        predict_time_sec=final_predict_time,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_timm_exact_image_dedup",
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_regularized_val_score": selected.get("regularized_val_score"),
            "selected_bias_mode": selected.get("bias_mode"),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
            "selected_label_policy": selected_spec.label_policy,
            "train_full_exact_image_groups": int(len(train_full_split.group_hashes)),
            "robot_test_exact_image_groups": int(len(test_split.group_hashes)),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        img.confusion_matrix(test_split.row_y, final_pred, labels=img.LABELS),
        index=img.CLASS_NAMES,
        columns=img.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "image_path", "label", "y", "segment_group", "specimen_group", "source", "image_hash"]
    ].copy()
    prediction_frame["pred_y"] = final_pred
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(img.ID2LABEL)
    for class_id, class_name in img.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "image_only_timm_exact_image_dedup_no_test_until_lock",
            "selected": selection_summary,
            "model": final_model,
            "selected_bias": selected_bias,
            "label_map": img.LABEL_MAP,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": "image_only_timm_exact_image_dedup_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "test_feature_timing": test_feature_timing,
        "final_test_report": final_row,
        "artifacts": {
            "val_leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "model_bundle": str(bundle_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    img.write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)

    print("\nFinal robot/test result after frozen image-only timm exact-image dedup selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_regularized_val_score",
                "selected_label_policy",
                "selected_bias_mode",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("Val leaderboard:", leaderboard_path.resolve(), flush=True)
    print("Selection lock:", selection_path.resolve(), flush=True)
    print("Final test report:", final_report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
