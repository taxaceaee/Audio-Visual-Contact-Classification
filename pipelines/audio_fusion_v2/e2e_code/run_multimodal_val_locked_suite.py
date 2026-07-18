from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


LABELS = np.arange(4, dtype=np.int64)
CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]
PROBA_COLUMNS = [f"proba_{name}" for name in CLASS_NAMES]
IMAGE_BIAS = np.asarray([0.0, 0.0, -1.2, -1.0], dtype=np.float64)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def calibrate(p: np.ndarray, gamma: float) -> np.ndarray:
    return normalize(np.exp(gamma * np.log(normalize(p))))


def apply_bias(p: np.ndarray) -> np.ndarray:
    return normalize(np.exp(np.log(normalize(p)) + IMAGE_BIAS[None, :]))


def audio_keys(files: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    group = files.astype(str).map(lambda x: re.sub(r"_window_\d+.*$", "", Path(x).stem))
    specimen = files.astype(str).map(lambda x: re.sub(r"_segment_.*$", "", Path(x).stem))
    codes, _ = pd.factorize(group, sort=False)
    seg = pd.DataFrame({"group": group, "specimen": specimen}).drop_duplicates("group")
    specimen_codes, _ = pd.factorize(seg["specimen"], sort=False)
    return codes.astype(np.int64), specimen_codes.astype(np.int64)


def segment_lift(files: pd.Series, window_p: np.ndarray) -> np.ndarray:
    codes, specimen_codes = audio_keys(files)
    n_groups = int(codes.max()) + 1
    logp = np.log(normalize(window_p))
    sums = np.vstack([np.bincount(codes, weights=logp[:, c], minlength=n_groups) for c in LABELS]).T
    seg = normalize(np.exp(sums - sums.max(axis=1, keepdims=True)))
    out = seg.copy()
    contact_mass = out[:, 1:4].sum(axis=1)
    contact_dist = normalize(out[:, 1:4])
    for sid in np.unique(specimen_codes):
        idx = np.where(specimen_codes == sid)[0]
        contact_idx = idx[contact_mass[idx] >= 0.45]
        if len(contact_idx) < 1:
            continue
        consensus = normalize(np.mean(contact_dist[contact_idx], axis=0, keepdims=True))[0]
        out[idx, 1:4] = contact_mass[idx, None] * consensus[None, :]
        out[idx, 0] = 1.0 - contact_mass[idx]
        if float(np.max(consensus)) < 0.45:
            continue
        pred_before = out[idx].argmax(axis=1)
        lift_idx = idx[(pred_before == 0) & (contact_mass[idx] >= 0.35)]
        if len(lift_idx):
            mass = np.clip(np.maximum(contact_mass[lift_idx], 0.58), 1e-12, 0.98)
            out[lift_idx, 0] = 1.0 - mass
            out[lift_idx, 1:4] = mass[:, None] * consensus[None, :]
    return normalize(out)[codes]


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        path = Path(str(row.image_path))
        try:
            image = Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", (224, 224))
        except Exception:
            image = Image.new("RGB", (224, 224))
        return self.transform(image), int(row.y)


def load_manifest(csv_path: Path, dataset_dir: Path, source: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    labels = {name: i for i, name in enumerate(CLASS_NAMES)}
    frame = pd.DataFrame({
        "audio_file": raw["audio_file"].astype(str),
        "image_file": raw["image_file"].astype(str),
        "image_path": raw["image_file"].map(lambda x: str(dataset_dir / x)),
        "label": raw["category"].astype(str).str.lower(),
        "source": source,
    })
    frame = frame[frame.label.isin(labels)].copy()
    frame["y"] = frame["label"].map(labels).astype(np.int64)
    return frame.reset_index(drop=True)


def specimen_group(files: pd.Series) -> pd.Series:
    return files.astype(str).map(lambda x: re.sub(r"_segment_.*$", "", Path(x).stem))


def build_group_split(root: Path, report: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = load_manifest(root / "audio_visual_dataset_default" / "dataset.csv", root / "audio_visual_dataset_default", "hand_train")
    frame["specimen_group"] = specimen_group(frame["audio_file"])
    indices = np.arange(len(frame))
    y = frame.y.to_numpy(dtype=np.int64)
    groups = frame.specimen_group.to_numpy()
    full_dist = np.bincount(y, minlength=4) / len(y)
    best = None
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in splitter.split(indices, y, groups):
        val_dist = np.bincount(y[val_idx], minlength=4) / max(len(val_idx), 1)
        score = abs(len(val_idx) / len(frame) - 0.2) + float(np.abs(val_dist - full_dist).sum())
        candidate = (score, train_idx, val_idx)
        if best is None or score < best[0]:
            best = candidate
    assert best is not None
    _, train_idx, val_idx = best
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    overlap = train_groups & val_groups
    if overlap:
        raise AssertionError(f"specimen-group leakage: {sorted(overlap)[:5]}")
    split_dir = report / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    frame.iloc[train_idx].to_csv(split_dir / "hand_group_train.csv", index=False)
    frame.iloc[val_idx].to_csv(split_dir / "hand_group_val.csv", index=False)
    summary = {
        "group_column": "specimen_group",
        "group_rule": "audio stem with _segment_... removed",
        "n_rows": int(len(frame)),
        "n_groups": int(frame.specimen_group.nunique()),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "group_overlap": int(len(overlap)),
        "train_label_counts": frame.iloc[train_idx].y.value_counts().sort_index().to_dict(),
        "val_label_counts": frame.iloc[val_idx].y.value_counts().sort_index().to_dict(),
    }
    (split_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frame, train_idx, val_idx


def image_cache(output: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    base = output / "image_deep_features" / "resnet18_224" / split_name
    return np.load(base / "X.npy"), np.load(base / "y.npy").astype(np.int64)


def image_model_candidates(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> dict[str, np.ndarray]:
    models = {
        "image_logreg_c01_balanced": LogisticRegression(C=0.1, class_weight="balanced", max_iter=2500, random_state=42),
        "image_logreg_c1_balanced": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2500, random_state=42),
        "image_extratrees": ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    out = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        raw = model.predict_proba(X_val)
        classes = np.asarray(model.classes_, dtype=np.int64)
        p = np.zeros((len(X_val), 4), dtype=np.float64)
        for col, cls in enumerate(classes):
            p[:, int(cls)] = raw[:, col]
        out[name] = apply_bias(p)
    return out


def image_predict(frame: pd.DataFrame, checkpoint: Path, device: torch.device) -> np.ndarray:
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 4))
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
    model.to(device).eval()
    loader = DataLoader(ImageDataset(frame), batch_size=64, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    rows = []
    with torch.inference_mode():
        for images, _ in loader:
            rows.append(model(images.to(device)).detach().cpu().numpy())
    logits = np.concatenate(rows, axis=0)
    return apply_bias(torch.softmax(torch.from_numpy(logits), dim=1).numpy())


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    by = (y > 0).astype(np.int64)
    bp = (pred > 0).astype(np.int64)
    return {
        "accuracy_4class": float(accuracy_score(y, pred)),
        "macro_precision_4class": float(precision_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall_4class": float(recall_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1_4class": float(f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0)),
        "binary_accuracy_ambient_noambient": float(accuracy_score(by, bp)),
        "binary_macro_precision_ambient_noambient": float(precision_score(by, bp, average="macro", zero_division=0)),
        "binary_macro_recall_ambient_noambient": float(recall_score(by, bp, average="macro", zero_division=0)),
        "binary_macro_f1_ambient_noambient": float(f1_score(by, bp, average="macro", zero_division=0)),
    }


def candidate_probability(name: str, pa: np.ndarray, pi: np.ndarray, params: dict) -> np.ndarray:
    if name == "linear":
        return normalize(params["alpha"] * pa + (1.0 - params["alpha"]) * pi)
    if name == "log":
        return normalize(np.exp(params["alpha"] * np.log(normalize(pa)) + (1.0 - params["alpha"]) * np.log(normalize(pi))))
    if name == "class_linear":
        weights = np.asarray(params["audio_class_weights"], dtype=np.float64)
        return normalize(weights[None, :] * pa + (1.0 - weights[None, :]) * pi)
    if name == "temperature_linear":
        qa = calibrate(pa, params["audio_gamma"])
        qi = calibrate(pi, params["image_gamma"])
        return normalize(params["alpha"] * qa + (1.0 - params["alpha"]) * qi)
    if name == "contact_route":
        alpha = params["alpha"]
        binary = normalize(alpha * pa[:, [0, 1, 2, 3]].copy() + (1 - alpha) * pi)
        contact_mass = binary[:, 1:4].sum(axis=1)
        contact_dist = normalize(alpha * normalize(pa)[:, 1:4] + (1 - alpha) * normalize(pi)[:, 1:4])
        out = np.zeros_like(binary)
        out[:, 0] = 1.0 - contact_mass
        out[:, 1:4] = contact_mass[:, None] * contact_dist
        return normalize(out)
    if name == "confidence_gate":
        ca = pa.max(axis=1)
        ci = pi.max(axis=1)
        use_image = (ci >= params["image_threshold"]) & (ci > ca + params["margin"])
        return np.where(use_image[:, None], pi, pa)
    raise KeyError(name)


def load_audio_oof(root: Path, output: Path) -> tuple[pd.DataFrame, np.ndarray]:
    raw = pd.read_csv(root / "audio_visual_dataset_default" / "dataset.csv")
    labels = {name: i for i, name in enumerate(CLASS_NAMES)}
    frame = pd.DataFrame({"audio_file": raw.audio_file.astype(str), "y": raw.category.astype(str).str.lower().map(labels).astype(np.int64)})
    run = output / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select"
    lock = json.loads((run / "reports" / "audio_highsr_temporal_tta_select_selected_without_test.json").read_text())
    cand = lock["selected_candidate"]
    w = lock["selected_tta_weights"]
    high = normalize(float(w["clean"]) * np.load(run / "oof_proba" / cand / "clean_oof_proba.npy") + float(w["robot_mix"]) * np.load(run / "oof_proba" / cand / "robot_mix_oof_proba.npy"))
    pair = normalize(np.load(output / "audio_feature_benchmarks" / "audio_group_consistency_pair_blend_select" / "oof_sources" / "pairwise_selected_clean_oof_proba.npy"))
    return frame, segment_lift(frame.audio_file, normalize(0.8 * high + 0.2 * pair))


def load_audio_test_after_lock(output: Path) -> tuple[pd.DataFrame, np.ndarray]:
    path = output / "audio_feature_benchmarks" / "audio_lift_source_blend_select" / "reports" / "audio_lift_source_blend_select_final_test_predictions.csv"
    frame = pd.read_csv(path)
    return frame[["audio_file", "y"]], normalize(frame[PROBA_COLUMNS].to_numpy())


def select_and_lock(root: Path, output: Path, report: Path) -> dict:
    image_run = output / "audio_feature_benchmarks" / "image_dl_finetune_resnet18_specimen_select"
    full_frame, train_idx, val_idx = build_group_split(root, report)
    val = full_frame.iloc[val_idx].copy()
    audio_frame, pa = load_audio_oof(root, output)
    X_full, y_cache = image_cache(output, "hand_train_full")
    if not np.array_equal(y_cache, full_frame.y.to_numpy(dtype=np.int64)):
        raise AssertionError("Image cache labels are not aligned with hand manifest")
    image_probs = image_model_candidates(X_full[train_idx], y_cache[train_idx], X_full[val_idx])
    merged = val[["audio_file", "y"]].merge(audio_frame[["audio_file", "y"]], on="audio_file", suffixes=("_image", "_audio"), validate="one_to_one")
    order = merged["audio_file"].to_numpy()
    index = pd.Series(audio_frame.audio_file.to_numpy()).reset_index().set_index(0)["index"]
    pa = pa[index.loc[order].to_numpy()]
    y = merged.y_image.to_numpy(dtype=np.int64)
    candidates = []
    for image_model_name, pi in image_probs.items():
        for alpha in np.linspace(0.0, 1.0, 21):
            for kind in ["linear", "log"]:
                candidates.append((kind, {"alpha": float(alpha), "image_model": image_model_name}, pi))
    for weights in [(0.25, 0.5, 0.75), (0.4, 0.6), (0.5,)]:
        if len(weights) == 3:
            for a in weights:
                for l in weights:
                    for t in weights:
                        for tw in weights:
                            candidates.append(("class_linear", {"audio_class_weights": [a, l, t, tw], "image_model": "image_extratrees"}, image_probs["image_extratrees"]))
    for alpha in [0.25, 0.5, 0.75]:
        for ag in [0.75, 1.0, 1.25]:
            for ig in [0.75, 1.0, 1.25]:
                for image_model_name, pi in image_probs.items():
                    candidates.append(("temperature_linear", {"alpha": alpha, "audio_gamma": ag, "image_gamma": ig, "image_model": image_model_name}, pi))
        for image_model_name, pi in image_probs.items():
            candidates.append(("contact_route", {"alpha": alpha, "image_model": image_model_name}, pi))
    for threshold in [0.55, 0.65, 0.75, 0.85]:
        for margin in [0.0, 0.05, 0.10, 0.15]:
            for image_model_name, pi in image_probs.items():
                candidates.append(("confidence_gate", {"image_threshold": threshold, "margin": margin, "image_model": image_model_name}, pi))
    rows = []
    for kind, params, pi in candidates:
        pred = candidate_probability(kind, pa, pi, params).argmax(axis=1)
        row = {"kind": kind, "params": json.dumps(params), **metrics(y, pred)}
        rows.append(row)
    leaderboard = pd.DataFrame(rows).sort_values(["macro_f1_4class", "binary_macro_f1_ambient_noambient", "accuracy_4class"], ascending=False).reset_index(drop=True)
    best = leaderboard.iloc[0].to_dict()
    lock = {"protocol": "multimodal_val_only_selection_then_robot_test", "selection_split": "hand_val_specimen_grouped", "n_selection": int(len(y)), "candidate_count": int(len(candidates)), "selected_kind": best["kind"], "selected_params": json.loads(best["params"]), "selection_metrics": {k: best[k] for k in ["accuracy_4class", "macro_precision_4class", "macro_recall_4class", "macro_f1_4class", "binary_macro_f1_ambient_noambient"]}, "test_data_loaded_before_lock": False, "group_overlap": 0}
    (report / "selection_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    leaderboard.to_csv(report / "hand_val_candidate_leaderboard.csv", index=False)
    return lock


def final_test(root: Path, output: Path, report: Path, lock: dict) -> dict:
    test = load_manifest(root / "audio_visual_dataset_robo_default" / "dataset.csv", root / "audio_visual_dataset_robo_default", "robot_test")
    full_frame = load_manifest(root / "audio_visual_dataset_default" / "dataset.csv", root / "audio_visual_dataset_default", "hand_train")
    X_full, y_cache = image_cache(output, "hand_train_full")
    if not np.array_equal(y_cache, full_frame.y.to_numpy(dtype=np.int64)):
        raise AssertionError("Image cache labels are not aligned with hand manifest")
    X_test, y_test_cache = image_cache(output, "robot_test")
    image_name = lock["selected_params"]["image_model"]
    image_models = {
        "image_logreg_c01_balanced": LogisticRegression(C=0.1, class_weight="balanced", max_iter=2500, random_state=42),
        "image_logreg_c1_balanced": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2500, random_state=42),
        "image_extratrees": ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    image_model = image_models[image_name].fit(X_full, y_cache)
    raw_image = image_model.predict_proba(X_test)
    pi = np.zeros((len(X_test), 4), dtype=np.float64)
    for col, cls in enumerate(image_model.classes_):
        pi[:, int(cls)] = raw_image[:, col]
    pi = apply_bias(pi)
    audio_frame, pa = load_audio_test_after_lock(output)
    merged = test[["audio_file", "y"]].merge(audio_frame, on="audio_file", suffixes=("_image", "_audio"), validate="one_to_one")
    order = merged.audio_file.to_numpy()
    idx = pd.Series(test.audio_file.to_numpy()).reset_index().set_index(0)["index"]
    pi = pi[idx.loc[order].to_numpy()]
    y = merged.y_image.to_numpy(dtype=np.int64)
    pred_p = candidate_probability(lock["selected_kind"], pa, pi, lock["selected_params"])
    pred = pred_p.argmax(axis=1).astype(np.int64)
    result = {"split": "robot_test_final", "n": int(len(y)), "selected_kind": lock["selected_kind"], "selected_params": lock["selected_params"], **metrics(y, pred), "per_class_4class": classification_report(y, pred, labels=LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0), "per_class_binary": classification_report((y > 0).astype(int), (pred > 0).astype(int), labels=[0, 1], target_names=["ambient", "noambient"], output_dict=True, zero_division=0), "confusion_matrix_4class": confusion_matrix(y, pred, labels=LABELS).tolist(), "confusion_matrix_binary": confusion_matrix((y > 0).astype(int), (pred > 0).astype(int), labels=[0, 1]).tolist()}
    (report / "final_test_metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(report / "final_test_report.csv", index=False)
    pd.DataFrame(result["confusion_matrix_4class"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(report / "final_test_confusion_matrix_4class.csv")
    pd.DataFrame(result["confusion_matrix_binary"], index=["ambient", "noambient"], columns=["ambient", "noambient"]).to_csv(report / "final_test_confusion_matrix_binary.csv")
    return result


def main() -> None:
    root = Path("/home/ttung05/Desktop/tree_base/tree_structures")
    output = Path("outputs")
    report = output / "audio_feature_benchmarks" / "multimodal_val_locked_suite"
    report.mkdir(parents=True, exist_ok=True)
    lock = select_and_lock(root, output, report)
    print("SELECTION LOCK:", json.dumps(lock, indent=2), flush=True)
    result = final_test(root, output, report, lock)
    print("FINAL TEST:", json.dumps(result, indent=2, default=float), flush=True)
    print("REPORT_DIR:", report.resolve(), flush=True)


if __name__ == "__main__":
    main()
