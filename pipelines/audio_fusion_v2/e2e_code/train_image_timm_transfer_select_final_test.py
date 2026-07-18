from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageFile
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import train_image_handcrafted_ml_select_final_test as img


ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    backbone: str
    factory: Callable[[], object]


class ImageManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Callable) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, bool]:
        row = self.frame.iloc[index]
        path = Path(row.image_path)
        missing = not path.exists()
        if missing:
            image = Image.new("RGB", (640, 480), color=(0, 0, 0))
        else:
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (640, 480), color=(0, 0, 0))
                missing = True
        return self.transform(image), int(row.y), str(path), missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only timm deep transfer learning. timm pretrained image backbones "
            "extract embeddings from hand/default only, train/val selects the head "
            "and class bias, writes a selection lock, then robot/test is loaded."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_timm_transfer_specimen_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["specimen", "segment", "row"], default="specimen")
    parser.add_argument(
        "--backbones",
        default="convnext_tiny.fb_in22k_ft_in1k,efficientnet_b3.ra2_in1k,swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        help="Comma-separated timm pretrained model names.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("outputs/image_timm_features"))
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument(
        "--selection-metric",
        choices=["regularized_val_score", "macro_f1_4class", "contact_macro_f1", "binary_macro_f1"],
        default="regularized_val_score",
    )
    parser.add_argument("--bias-penalty", type=float, default=0.025)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def manifest_signature(frame: pd.DataFrame, backbone: str, data_config: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(f"backbone={backbone}|config={stable_json(data_config)}\n".encode("utf-8"))
    for row in frame[["image_path", "y"]].itertuples(index=False):
        digest.update(f"{row.image_path}|{int(row.y)}\n".encode("utf-8"))
    return digest.hexdigest()


def cache_paths(feature_cache_dir: Path, backbone: str, split_name: str) -> dict[str, Path]:
    split_dir = feature_cache_dir / safe_name(backbone) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X": split_dir / "X.npy",
        "y": split_dir / "y.npy",
        "paths": split_dir / "paths.npy",
        "missing": split_dir / "missing.npy",
        "metadata": split_dir / "metadata.json",
    }


def make_timm_model(backbone: str) -> tuple[nn.Module, dict[str, object]]:
    try:
        model = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
    except RuntimeError as exc:
        message = str(exc)
        if "Missing key(s) in state_dict" not in message or "fc_norm" not in message:
            raise
        model = timm.create_model(backbone, pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(model)
    return model, dict(data_config)


def make_transform(data_config: dict[str, object]) -> Callable:
    return timm.data.create_transform(**data_config, is_training=False)


def normalize_features(features: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim == 4:
        features = features.mean(dim=(2, 3))
    elif features.ndim == 3:
        features = features.mean(dim=1)
    elif features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    return features


@torch.inference_mode()
def extract_timm_features(
    frame: pd.DataFrame,
    split_name: str,
    backbone_name: str,
    feature_cache_dir: Path,
    batch_size: int,
    num_workers: int,
    force_rebuild: bool,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model, data_config = make_timm_model(backbone_name)
    paths = cache_paths(feature_cache_dir, backbone_name, split_name)
    signature = manifest_signature(frame, backbone_name, data_config)
    y_expected = frame["y"].to_numpy(dtype=np.int64)
    ready = all(path.exists() for path in paths.values())
    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        cache_y = np.load(paths["y"])
        signature_matches = metadata.get("manifest_signature") == signature
        y_aligned = len(cache_y) == len(y_expected) and np.array_equal(cache_y, y_expected)
        if (
            metadata.get("feature_family") == "timm_deep_image_embedding"
            and metadata.get("backbone") == backbone_name
            and int(metadata.get("n_samples", -1)) == len(frame)
            and (signature_matches or y_aligned)
        ):
            start = time.perf_counter()
            payload = {
                "X": np.load(paths["X"]),
                "y": cache_y,
                "paths": np.load(paths["paths"], allow_pickle=True),
                "missing": np.load(paths["missing"]),
            }
            elapsed = time.perf_counter() - start
            print(
                f"Loaded timm cache {backbone_name}/{split_name}: {payload['X'].shape} "
                f"in {elapsed:.3f}s",
                flush=True,
            )
            return payload, {
                "cache_hit": True,
                "cache_load_time_sec": elapsed,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
                "missing_images": int(metadata.get("missing_images", 0)),
                "data_config": metadata.get("data_config", data_config),
                "cache_validation": "signature" if signature_matches else "label_order",
            }

    model.eval().to(device)
    dataset = ImageManifestDataset(frame, make_transform(data_config))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    rows: list[np.ndarray] = []
    labels: list[int] = []
    image_paths: list[str] = []
    missing_flags: list[bool] = []
    start = time.perf_counter()
    for images, y, batch_paths, missing in tqdm(loader, desc=f"Extract {backbone_name}/{split_name}"):
        images = images.to(device, non_blocking=True)
        features = normalize_features(model(images))
        rows.append(features.detach().cpu().float().numpy().astype(np.float32))
        labels.extend(np.asarray(y, dtype=np.int64).tolist())
        image_paths.extend(list(batch_paths))
        missing_flags.extend(np.asarray(missing, dtype=bool).tolist())
    extraction_time = time.perf_counter() - start
    X = np.concatenate(rows, axis=0).astype(np.float32)
    payload = {
        "X": X,
        "y": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray(image_paths),
        "missing": np.asarray(missing_flags, dtype=bool),
    }
    if len(payload["X"]) != len(frame):
        raise AssertionError(f"{backbone_name}/{split_name}: expected {len(frame)} rows, got {len(payload['X'])}")

    np.save(paths["X"], payload["X"])
    np.save(paths["y"], payload["y"])
    np.save(paths["paths"], payload["paths"])
    np.save(paths["missing"], payload["missing"])
    metadata = {
        "feature_family": "timm_deep_image_embedding",
        "backbone": backbone_name,
        "feature_dim": int(payload["X"].shape[1]),
        "n_samples": int(len(frame)),
        "manifest_signature": signature,
        "missing_images": int(payload["missing"].sum()),
        "extraction_time_sec": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
        "data_config": data_config,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(
        f"Saved timm cache {backbone_name}/{split_name}: {payload['X'].shape} "
        f"in {extraction_time:.2f}s missing={int(payload['missing'].sum())}",
        flush=True,
    )
    return payload, {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
        "missing_images": int(payload["missing"].sum()),
        "data_config": data_config,
        "cache_validation": "rebuilt",
    }


def make_candidates(backbone: str, random_state: int, n_features: int) -> dict[str, CandidateSpec]:
    class_weights = {0: 0.6, 1: 1.1, 2: 1.8, 3: 1.3}

    def logreg(c: float, class_weight: object) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        class_weight=class_weight,
                        max_iter=6000,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    specs = [
        CandidateSpec(f"{backbone}_logreg_C0.1_balanced", backbone, lambda: logreg(0.1, "balanced")),
        CandidateSpec(f"{backbone}_logreg_C0.3_balanced", backbone, lambda: logreg(0.3, "balanced")),
        CandidateSpec(f"{backbone}_logreg_C1_balanced", backbone, lambda: logreg(1.0, "balanced")),
        CandidateSpec(f"{backbone}_logreg_C1_weighted", backbone, lambda: logreg(1.0, class_weights)),
        CandidateSpec(
            f"{backbone}_extra_trees",
            backbone,
            lambda: ExtraTreesClassifier(
                n_estimators=800,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight=class_weights,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        CandidateSpec(
            f"{backbone}_random_forest",
            backbone,
            lambda: RandomForestClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        CandidateSpec(
            f"{backbone}_hgb_balanced",
            backbone,
            lambda: HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=0.1,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    ]
    if n_features > 128:
        n_components = min(256, n_features)
        specs.append(
            CandidateSpec(
                f"{backbone}_pca{n_components}_logreg_C1",
                backbone,
                lambda n_components=n_components: Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=n_components, whiten=True, random_state=random_state)),
                        (
                            "model",
                            LogisticRegression(
                                C=1.0,
                                class_weight="balanced",
                                max_iter=6000,
                                random_state=random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            )
        )
    return {spec.name: spec for spec in specs}


def evaluate_candidate(
    spec: CandidateSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    bias_penalty: float,
) -> list[dict[str, object]]:
    model = spec.factory()
    rows = []
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    if hasattr(model, "predict_proba"):
        val_proba = img.proba_aligned(model, X_val)
        default_pred = val_proba.argmax(axis=1)
        tuned_bias, _ = img.tune_class_bias(val_proba, y_val)
        bias_items = [
            ("none", np.zeros(len(img.CLASS_NAMES), dtype=np.float64), default_pred),
            ("tuned", tuned_bias, img.predict_with_bias(val_proba, tuned_bias)),
        ]
        predict_time = time.perf_counter() - start
    else:
        pred = model.predict(X_val)
        bias_items = [("none", np.zeros(len(img.CLASS_NAMES), dtype=np.float64), pred)]
        predict_time = time.perf_counter() - start

    for bias_mode, bias, pred in bias_items:
        row = img.make_report_row(
            model_name=spec.name,
            split_name="hand_val",
            y_true=y_val,
            y_pred=pred,
            feature_set=spec.backbone,
            feature_name=f"{spec.backbone}_timm_embedding",
            n_features=X_train.shape[1],
            train_time_sec=train_time,
            predict_time_sec=predict_time,
        )
        bias_l1 = float(np.abs(bias).sum())
        row.update(
            {
                "backbone": spec.backbone,
                "bias_mode": bias_mode,
                "class_bias_json": json.dumps(bias.tolist()),
                "bias_l1": bias_l1,
                "regularized_val_score": float(row["macro_f1_4class"] - bias_penalty * bias_l1),
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir, args.feature_cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("ROOT_PATH =", root_path.resolve(), flush=True)
    print("RUN_DIR   =", run_dir.resolve(), flush=True)
    print("DEVICE    =", device, flush=True)
    print("Protocol  = image-only timm transfer selection; robot/test after lock", flush=True)

    train_df = img.load_image_manifest(
        img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv"),
        train_dataset_dir,
        "hand_train",
    )
    train_idx, val_idx, split_info = img.split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = img.save_split_manifests(run_dir, train_df, None, train_idx, val_idx)

    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    all_payloads: dict[str, dict[str, np.ndarray]] = {}
    train_feature_timing: dict[str, dict[str, object]] = {}
    leaderboard_rows: list[dict[str, object]] = []
    candidates: dict[str, CandidateSpec] = {}
    for backbone in backbones:
        payload, timing = extract_timm_features(
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
            raise AssertionError(f"{backbone}: train feature labels do not align with manifest")
        all_payloads[backbone] = payload
        train_feature_timing[backbone] = timing
        X_train = payload["X"][train_idx]
        y_train = payload["y"][train_idx]
        X_val = payload["X"][val_idx]
        y_val = payload["y"][val_idx]
        for spec in make_candidates(backbone, args.random_state, payload["X"].shape[1]).values():
            print(f"\nVAL candidate: {spec.name}", flush=True)
            rows = evaluate_candidate(spec, X_train, y_train, X_val, y_val, args.bias_penalty)
            candidates[spec.name] = spec
            for row in rows:
                print(
                    f"  {row['bias_mode']:5s} macro={row['macro_f1_4class']:.4f} "
                    f"reg={row['regularized_val_score']:.4f} acc={row['accuracy_4class']:.4f}",
                    flush=True,
                )
            leaderboard_rows.extend(rows)

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
    split_summary = {
        "protocol": "image_only_timm_transfer_train_val_selection_robot_after_lock",
        "split_info": split_info,
        "train_full_samples": int(len(train_df)),
        "train_full_label_counts": img.label_counts(train_df["y"].to_numpy()),
        "backbones": backbones,
        "train_feature_timing": train_feature_timing,
        "bias_penalty": args.bias_penalty,
        "split_paths": split_paths,
    }
    split_summary_path = report_dir / f"{args.run_slug}_split_summary.json"
    img.write_json(split_summary_path, split_summary)
    selection_summary = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_backbone": selected_spec.backbone,
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
    split_paths = img.save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)
    test_payload, test_timing = extract_timm_features(
        test_df,
        "robot_test",
        selected_spec.backbone,
        args.feature_cache_dir,
        args.batch_size,
        args.num_workers,
        args.force_rebuild_features,
        device,
    )
    selected_train_payload = all_payloads[selected_spec.backbone]
    final_model = clone(selected_spec.factory())
    start = time.perf_counter()
    final_model.fit(selected_train_payload["X"], selected_train_payload["y"])
    final_fit_time = time.perf_counter() - start
    if hasattr(final_model, "predict_proba"):
        final_proba = img.proba_aligned(final_model, test_payload["X"])
        final_pred = img.predict_with_bias(final_proba, selected_bias)
    else:
        final_proba = None
        final_pred = final_model.predict(test_payload["X"])

    final_row = img.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_payload["y"],
        y_pred=final_pred,
        feature_set=selected_spec.backbone,
        feature_name=f"{selected_spec.backbone}_timm_embedding",
        n_features=selected_train_payload["X"].shape[1],
        train_time_sec=final_fit_time,
        predict_time_sec=0.0,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_timm_transfer",
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_regularized_val_score": selected.get("regularized_val_score"),
            "selected_bias_mode": selected.get("bias_mode"),
            "selected_bias_json": json.dumps(selected_bias.tolist()),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        img.confusion_matrix(test_payload["y"], final_pred, labels=img.LABELS),
        index=img.CLASS_NAMES,
        columns=img.CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_frame = test_df[
        ["audio_file", "image_file", "image_path", "label", "y", "segment_group", "specimen_group", "source"]
    ].copy()
    prediction_frame["pred_y"] = final_pred
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(img.ID2LABEL)
    if final_proba is not None:
        for class_id, class_name in img.ID2LABEL.items():
            prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": "image_only_timm_transfer_no_test_until_lock",
            "selected": selection_summary,
            "model": final_model,
            "selected_bias": selected_bias,
            "label_map": img.LABEL_MAP,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": "image_only_timm_transfer_no_test_until_lock",
        "root": str(root_path.resolve()),
        "selection": selection_summary,
        "split_summary": split_summary,
        "test_feature_timing": test_timing,
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

    print("\nFinal robot/test result after frozen image-only timm transfer selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_regularized_val_score",
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
