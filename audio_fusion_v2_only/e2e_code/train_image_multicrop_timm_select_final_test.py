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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import train_image_handcrafted_ml_select_final_test as img


ImageFile.LOAD_TRUNCATED_IMAGES = True
LABELS = img.LABELS
CLASS_NAMES = img.CLASS_NAMES
PROBA_COLUMNS = [f"proba_{img.ID2LABEL[index]}" for index in LABELS]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    backbone: str
    factory: Callable[[], object]


class MultiCropDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, crop_names: list[str], transform: Callable) -> None:
        self.frame = frame.reset_index(drop=True)
        self.crop_names = crop_names
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
        crops = [self.transform(crop_image(image, name)) for name in self.crop_names]
        return torch.stack(crops, dim=0), int(row.y), str(path), missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only multicrop timm transfer. Selection uses a group-safe "
            "hand/train validation split; robot/test is loaded only after the "
            "method card and selection lock are written."
        )
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--run-slug", default="image_multicrop_timm_specimen_select")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["specimen", "segment", "row"], default="specimen")
    parser.add_argument(
        "--backbones",
        default="vit_base_patch16_clip_224.openai_ft_in1k,convnext_tiny.fb_in22k_ft_in1k",
    )
    parser.add_argument(
        "--crops",
        default="full,center,bottom,lower_center",
        help="Comma-separated deterministic crops. Available: full,center,bottom,lower_center,upper,left,right.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("outputs/image_multicrop_timm_features"),
    )
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument(
        "--decoders",
        default="window,segment_mean,segment_log,specimen_mean,specimen_log",
    )
    parser.add_argument("--bias-values", default="-1.2,-0.8,-0.4,0.0,0.4,0.8,1.2")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def crop_image(image: Image.Image, crop_name: str) -> Image.Image:
    width, height = image.size
    if crop_name == "full":
        box = (0, 0, width, height)
    elif crop_name == "center":
        side = int(min(width, height) * 0.86)
        left = (width - side) // 2
        top = (height - side) // 2
        box = (left, top, left + side, top + side)
    elif crop_name == "bottom":
        box = (0, int(height * 0.35), width, height)
    elif crop_name == "lower_center":
        crop_w = int(width * 0.72)
        crop_h = int(height * 0.72)
        left = (width - crop_w) // 2
        top = height - crop_h
        box = (left, top, left + crop_w, height)
    elif crop_name == "upper":
        box = (0, 0, width, int(height * 0.70))
    elif crop_name == "left":
        box = (0, 0, int(width * 0.62), height)
    elif crop_name == "right":
        box = (int(width * 0.38), 0, width, height)
    else:
        raise KeyError(f"Unknown crop: {crop_name}")
    return image.crop(box)


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def manifest_signature(frame: pd.DataFrame, backbone: str, crop_names: list[str], data_config: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"backbone={backbone}|crops={','.join(crop_names)}|config={stable_json(data_config)}\n".encode("utf-8")
    )
    for row in frame[["image_path", "y"]].itertuples(index=False):
        digest.update(f"{row.image_path}|{int(row.y)}\n".encode("utf-8"))
    return digest.hexdigest()


def cache_paths(feature_cache_dir: Path, backbone: str, crop_names: list[str], split_name: str) -> dict[str, Path]:
    split_dir = feature_cache_dir / safe_name(backbone) / safe_name("-".join(crop_names)) / split_name
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
def extract_multicrop_features(
    frame: pd.DataFrame,
    split_name: str,
    backbone: str,
    crop_names: list[str],
    feature_cache_dir: Path,
    batch_size: int,
    num_workers: int,
    force_rebuild: bool,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model, data_config = make_timm_model(backbone)
    paths = cache_paths(feature_cache_dir, backbone, crop_names, split_name)
    signature = manifest_signature(frame, backbone, crop_names, data_config)
    y_expected = frame["y"].to_numpy(dtype=np.int64)
    ready = all(path.exists() for path in paths.values())
    if ready and not force_rebuild:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        cache_y = np.load(paths["y"])
        if (
            metadata.get("feature_family") == "timm_multicrop_image_embedding"
            and metadata.get("backbone") == backbone
            and metadata.get("crop_names") == crop_names
            and metadata.get("manifest_signature") == signature
            and len(cache_y) == len(y_expected)
            and np.array_equal(cache_y, y_expected)
        ):
            start = time.perf_counter()
            payload = {
                "X": np.load(paths["X"]),
                "y": cache_y,
                "paths": np.load(paths["paths"], allow_pickle=True),
                "missing": np.load(paths["missing"]),
            }
            elapsed = time.perf_counter() - start
            print(f"Loaded multicrop cache {backbone}/{split_name}: {payload['X'].shape} in {elapsed:.3f}s")
            return payload, {
                "cache_hit": True,
                "cache_load_time_sec": elapsed,
                "extraction_time_sec_current_run": 0.0,
                "extraction_time_sec_original": float(metadata["extraction_time_sec"]),
                "missing_images": int(metadata.get("missing_images", 0)),
                "data_config": metadata.get("data_config", data_config),
            }

    model.eval().to(device)
    dataset = MultiCropDataset(frame, crop_names, make_transform(data_config))
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
    for crop_batch, y, batch_paths, missing in tqdm(loader, desc=f"Extract {backbone}/{split_name}/multicrop"):
        batch_size_current, n_crops = crop_batch.shape[:2]
        flat = crop_batch.reshape(batch_size_current * n_crops, *crop_batch.shape[2:]).to(device, non_blocking=True)
        features = normalize_features(model(flat))
        features = features.reshape(batch_size_current, n_crops, -1)
        features = features.reshape(batch_size_current, -1)
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
    np.save(paths["X"], payload["X"])
    np.save(paths["y"], payload["y"])
    np.save(paths["paths"], payload["paths"])
    np.save(paths["missing"], payload["missing"])
    metadata = {
        "feature_family": "timm_multicrop_image_embedding",
        "backbone": backbone,
        "crop_names": crop_names,
        "feature_dim": int(payload["X"].shape[1]),
        "n_samples": int(len(frame)),
        "manifest_signature": signature,
        "missing_images": int(payload["missing"].sum()),
        "extraction_time_sec": extraction_time,
        "extraction_time_per_sample_sec": extraction_time / len(frame),
        "data_config": data_config,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Saved multicrop cache {backbone}/{split_name}: {payload['X'].shape} in {extraction_time:.2f}s")
    return payload, {
        "cache_hit": False,
        "cache_load_time_sec": 0.0,
        "extraction_time_sec_current_run": extraction_time,
        "extraction_time_sec_original": extraction_time,
        "missing_images": int(payload["missing"].sum()),
        "data_config": data_config,
    }


def make_candidates(backbone: str, random_state: int) -> dict[str, CandidateSpec]:
    return {
        f"{backbone}__logreg_c003_bal": CandidateSpec(
            name=f"{backbone}__logreg_c003_bal",
            backbone=backbone,
            factory=lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.03, class_weight="balanced", max_iter=3000, random_state=random_state),
            ),
        ),
        f"{backbone}__logreg_c01_bal": CandidateSpec(
            name=f"{backbone}__logreg_c01_bal",
            backbone=backbone,
            factory=lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=random_state),
            ),
        ),
        f"{backbone}__logreg_c03_bal": CandidateSpec(
            name=f"{backbone}__logreg_c03_bal",
            backbone=backbone,
            factory=lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000, random_state=random_state),
            ),
        ),
        f"{backbone}__extra_trees": CandidateSpec(
            name=f"{backbone}__extra_trees",
            backbone=backbone,
            factory=lambda: ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        f"{backbone}__random_forest": CandidateSpec(
            name=f"{backbone}__random_forest",
            backbone=backbone,
            factory=lambda: RandomForestClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
    }


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def apply_bias(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    scores = np.log(np.clip(proba, 1e-12, 1.0)) + bias.reshape(1, -1)
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def group_decode(frame: pd.DataFrame, proba: np.ndarray, decoder: str) -> np.ndarray:
    if decoder == "window":
        return normalize_proba(proba)
    if decoder.startswith("segment"):
        groups = frame["segment_group"].astype(str).to_numpy()
    elif decoder.startswith("specimen"):
        groups = frame["specimen_group"].astype(str).to_numpy()
    else:
        raise KeyError(f"Unknown decoder: {decoder}")
    group_codes, _ = pd.factorize(groups, sort=False)
    n_groups = int(group_codes.max()) + 1
    if decoder.endswith("_log"):
        values = np.log(np.clip(proba, 1e-12, 1.0))
        pooled = np.vstack(
            [np.bincount(group_codes, weights=values[:, class_id], minlength=n_groups) for class_id in LABELS]
        ).T
        pooled = np.exp(pooled - pooled.max(axis=1, keepdims=True))
        pooled = pooled / pooled.sum(axis=1, keepdims=True)
    elif decoder.endswith("_mean"):
        pooled = np.vstack(
            [np.bincount(group_codes, weights=proba[:, class_id], minlength=n_groups) for class_id in LABELS]
        ).T
        counts = np.bincount(group_codes, minlength=n_groups).astype(np.float64)
        pooled = pooled / counts[:, None]
    else:
        raise KeyError(f"Unknown decoder: {decoder}")
    return normalize_proba(pooled[group_codes])


def tune_bias_and_decoder(
    frame: pd.DataFrame,
    proba: np.ndarray,
    y_true: np.ndarray,
    decoders: list[str],
    bias_values: np.ndarray,
) -> tuple[np.ndarray, str, np.ndarray, float]:
    best_score = -1.0
    best_bias = np.zeros(len(LABELS), dtype=np.float64)
    best_decoder = "window"
    best_proba = normalize_proba(proba)
    for b1 in bias_values:
        for b2 in bias_values:
            for b3 in bias_values:
                bias = np.asarray([0.0, b1, b2, b3], dtype=np.float64)
                biased = apply_bias(proba, bias)
                for decoder in decoders:
                    decoded = group_decode(frame, biased, decoder)
                    pred = decoded.argmax(axis=1)
                    score = f1_score(y_true, pred, average="macro", zero_division=0)
                    if score > best_score:
                        best_score = float(score)
                        best_bias = bias
                        best_decoder = decoder
                        best_proba = decoded
    return best_bias, best_decoder, best_proba, best_score


def evaluate_candidate(
    spec: CandidateSpec,
    train_payload: dict[str, np.ndarray],
    val_payload: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_df: pd.DataFrame,
    decoders: list[str],
    bias_values: np.ndarray,
) -> tuple[dict, np.ndarray, object]:
    X_train = train_payload["X"][train_idx]
    y_train = train_payload["y"][train_idx]
    X_val = val_payload["X"][val_idx]
    y_val = val_payload["y"][val_idx]
    val_frame = train_df.iloc[val_idx].reset_index(drop=True)
    print(f"\nVAL candidate: {spec.name}")
    model = spec.factory()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    val_proba = normalize_proba(img.proba_aligned(model, X_val))
    predict_time = time.perf_counter() - start
    bias, decoder, decoded_proba, tuned_macro_f1 = tune_bias_and_decoder(
        val_frame,
        val_proba,
        y_val,
        decoders,
        bias_values,
    )
    pred = decoded_proba.argmax(axis=1)
    row = img.make_report_row(
        model_name=spec.name,
        split_name="hand_val",
        y_true=y_val,
        y_pred=pred,
        feature_set=spec.backbone,
        feature_name=f"{spec.backbone}_multicrop",
        n_features=train_payload["X"].shape[1],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
    )
    row.update(
        {
            "backbone": spec.backbone,
            "decoder": decoder,
            "class_bias_json": json.dumps(bias.tolist()),
            "selected_val_macro_f1": tuned_macro_f1,
            "raw_window_macro_f1": f1_score(y_val, val_proba.argmax(axis=1), average="macro", zero_division=0),
        }
    )
    print(
        f"  raw={row['raw_window_macro_f1']:.4f} tuned={row['macro_f1_4class']:.4f} "
        f"decoder={decoder} acc={row['accuracy_4class']:.4f} fit={train_time:.2f}s"
    )
    return row, val_proba, model


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    root_path = img.resolve_root(args.root)
    train_dataset_dir = root_path / "audio_visual_dataset_default"
    test_dataset_dir = root_path / "audio_visual_dataset_robo_default"
    run_dir = args.output / "audio_feature_benchmarks" / args.run_slug
    report_dir = run_dir / "reports"
    model_dir = run_dir / "models"
    for directory in [run_dir, report_dir, model_dir, args.feature_cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    backbones = [value.strip() for value in args.backbones.split(",") if value.strip()]
    crop_names = [value.strip() for value in args.crops.split(",") if value.strip()]
    decoders = [value.strip() for value in args.decoders.split(",") if value.strip()]
    bias_values = np.asarray([float(value) for value in args.bias_values.split(",")], dtype=np.float64)

    print("ROOT_PATH =", root_path.resolve(), flush=True)
    print("RUN_DIR   =", run_dir.resolve(), flush=True)
    print("DEVICE    =", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    print("Protocol  = image-only multicrop timm transfer; robot/test after selection lock", flush=True)
    print("Backbones =", backbones, flush=True)
    print("Crops     =", crop_names, flush=True)

    train_csv = img.require_file(train_dataset_dir / "dataset.csv", "hand/default dataset.csv")
    train_df = img.load_image_manifest(train_csv, train_dataset_dir, "hand_train")
    train_idx, val_idx, split_info = img.split_train_val(
        train_df,
        val_size=args.val_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )
    split_paths = img.save_split_manifests(run_dir, train_df, None, train_idx, val_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    leaderboard_rows: list[dict] = []
    candidate_specs: dict[str, CandidateSpec] = {}
    feature_timings: dict[str, dict[str, object]] = {}
    for backbone in backbones:
        payload, timing = extract_multicrop_features(
            train_df,
            "hand_train_full",
            backbone,
            crop_names,
            args.feature_cache_dir,
            args.batch_size,
            args.num_workers,
            args.force_rebuild_features,
            device,
        )
        if not np.array_equal(payload["y"], train_df["y"].to_numpy(dtype=np.int64)):
            raise AssertionError(f"{backbone}: train labels are not aligned")
        feature_timings[backbone] = timing
        for spec in make_candidates(backbone, args.random_state).values():
            row, _, _ = evaluate_candidate(
                spec,
                payload,
                payload,
                train_idx,
                val_idx,
                train_df,
                decoders,
                bias_values,
            )
            leaderboard_rows.append(row)
            candidate_specs[spec.name] = spec

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["macro_f1_4class", "contact_macro_f1", "binary_macro_f1", "accuracy_4class"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_path = report_dir / f"{args.run_slug}_val_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    selected = leaderboard.iloc[0].to_dict()
    selected_name = str(selected["model"])
    selected_spec = candidate_specs[selected_name]
    selected_bias = np.asarray(json.loads(selected["class_bias_json"]), dtype=np.float64)
    selected_decoder = str(selected["decoder"])

    method_card = {
        "protocol": "image_only_multicrop_timm_train_val_selection_robot_after_lock",
        "allowed_selection_data": "hand/default train labels, grouped hand validation, image pixels only",
        "forbidden_selection_data": "robot/test labels or predictions before selection lock, audio features, multimodal features, filename label tokens as predictive features",
        "split_info": split_info,
        "crop_names": crop_names,
        "decoders": decoders,
        "backbones": backbones,
        "feature_timings": feature_timings,
        "split_paths": split_paths,
    }
    method_card_path = report_dir / f"{args.run_slug}_method_card_before_test.json"
    write_json(method_card_path, method_card)
    selection = {
        "selected_without_test": selected,
        "selected_model": selected_name,
        "selected_backbone": selected_spec.backbone,
        "selected_decoder": selected_decoder,
        "selected_bias": selected_bias.tolist(),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "method_card": str(method_card_path.resolve()),
    }
    selection_path = report_dir / f"{args.run_slug}_selected_without_test.json"
    write_json(selection_path, selection)
    print("\nSelection lock written before loading robot/test:", flush=True)
    print(json.dumps(selection, indent=2, default=float), flush=True)

    test_csv = img.require_file(test_dataset_dir / "dataset.csv", "robot dataset.csv")
    test_df = img.load_image_manifest(test_csv, test_dataset_dir, "robot_test")
    split_paths = img.save_split_manifests(run_dir, train_df, test_df, train_idx, val_idx)
    train_payload, _ = extract_multicrop_features(
        train_df,
        "hand_train_full",
        selected_spec.backbone,
        crop_names,
        args.feature_cache_dir,
        args.batch_size,
        args.num_workers,
        False,
        device,
    )
    test_payload, test_timing = extract_multicrop_features(
        test_df,
        "robot_test",
        selected_spec.backbone,
        crop_names,
        args.feature_cache_dir,
        args.batch_size,
        args.num_workers,
        args.force_rebuild_features,
        device,
    )
    model = clone(selected_spec.factory())
    start = time.perf_counter()
    model.fit(train_payload["X"], train_payload["y"])
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    raw_test_proba = normalize_proba(img.proba_aligned(model, test_payload["X"]))
    biased_test_proba = apply_bias(raw_test_proba, selected_bias)
    final_proba = group_decode(test_df, biased_test_proba, selected_decoder)
    predict_time = time.perf_counter() - start
    final_pred = final_proba.argmax(axis=1)
    final_row = img.make_report_row(
        model_name=selected_name,
        split_name="robot_test_final",
        y_true=test_payload["y"],
        y_pred=final_pred,
        feature_set=selected_spec.backbone,
        feature_name=f"{selected_spec.backbone}_multicrop",
        n_features=train_payload["X"].shape[1],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
    )
    final_row.update(
        {
            "selected_by": "hand_train_val_image_only_multicrop_timm",
            "selected_val_macro_f1": selected.get("macro_f1_4class"),
            "selected_val_contact_macro_f1": selected.get("contact_macro_f1"),
            "selected_val_binary_macro_f1": selected.get("binary_macro_f1"),
            "selected_decoder": selected_decoder,
            "selected_bias_json": json.dumps(selected_bias.tolist()),
            "selected_crops_json": json.dumps(crop_names),
        }
    )
    final_report_path = report_dir / f"{args.run_slug}_final_test_report.csv"
    pd.DataFrame([final_row]).to_csv(final_report_path, index=False)
    confusion_path = report_dir / f"{args.run_slug}_final_test_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(test_payload["y"], final_pred, labels=LABELS),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(confusion_path)
    predictions_path = report_dir / f"{args.run_slug}_final_test_predictions.csv"
    prediction_columns = [
        "audio_file",
        "image_file",
        "image_path",
        "label",
        "y",
        "segment_group",
        "specimen_group",
        "source",
    ]
    prediction_frame = test_df[prediction_columns].copy()
    prediction_frame["pred_y"] = final_pred.astype(int)
    prediction_frame["pred_label"] = prediction_frame["pred_y"].map(img.ID2LABEL)
    for class_id, class_name in img.ID2LABEL.items():
        prediction_frame[f"proba_{class_name}"] = final_proba[:, class_id]
    prediction_frame.to_csv(predictions_path, index=False)

    bundle_path = model_dir / f"{args.run_slug}_selected_model_bundle.joblib"
    joblib.dump(
        {
            "protocol": method_card["protocol"],
            "method_card": method_card,
            "selection": selection,
            "model": model,
            "selected_bias": selected_bias,
            "selected_decoder": selected_decoder,
            "crop_names": crop_names,
        },
        bundle_path,
    )
    protocol_summary = {
        "protocol": method_card["protocol"],
        "root": str(root_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "method_card": method_card,
        "selection": selection,
        "test_feature_timing": test_timing,
        "final_test_report": final_row,
        "artifacts": {
            "method_card": str(method_card_path.resolve()),
            "leaderboard": str(leaderboard_path.resolve()),
            "selection_lock": str(selection_path.resolve()),
            "final_test_report": str(final_report_path.resolve()),
            "final_test_confusion_matrix": str(confusion_path.resolve()),
            "final_test_predictions": str(predictions_path.resolve()),
            "selected_model_bundle": str(bundle_path.resolve()),
            "split_manifests": split_paths,
        },
    }
    write_json(report_dir / f"{args.run_slug}_protocol_summary.json", protocol_summary)
    print("\nFinal robot/test result after frozen image-only multicrop timm selection:", flush=True)
    print(
        pd.DataFrame([final_row])[
            [
                "accuracy_4class",
                "macro_f1_4class",
                "contact_macro_f1",
                "binary_macro_f1",
                "selected_val_macro_f1",
                "selected_decoder",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("Final test report:", final_report_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
