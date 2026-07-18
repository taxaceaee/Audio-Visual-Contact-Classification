from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from PIL import Image, ImageFile
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torch.utils.data import Dataset


LABELS = np.arange(4, dtype=np.int64)
CLASS_NAMES = ["ambient", "leaf", "trunk", "twig"]
PROBA_COLUMNS = [f"proba_{name}" for name in CLASS_NAMES]
IMAGE_BIAS = np.asarray([0.0, 0.0, -1.2, -1.0], dtype=np.float64)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(str(row.image_path))
        try:
            image = Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", (224, 224))
        except Exception:
            image = Image.new("RGB", (224, 224))
        return self.transform(image), int(row.y)


def load_image_manifest(csv_path: Path, dataset_dir: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    out = pd.DataFrame({
        "audio_file": frame["audio_file"].astype(str),
        "image_file": frame["image_file"].astype(str),
        "image_path": frame["image_file"].map(lambda x: str(dataset_dir / x)),
        "label": frame["category"].astype(str).str.lower(),
        "source": source,
    })
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    out = out[out.label.isin(label_map)].copy()
    out["y"] = out["label"].map(label_map).astype(np.int64)
    return out.reset_index(drop=True)


def resolve_root() -> Path:
    candidates = [Path("tree_structures"), Path("/home/ttung05/Desktop/tree_base/tree_structures")]
    for candidate in candidates:
        if (candidate / "audio_visual_dataset_default" / "dataset.csv").exists():
            return candidate
    raise FileNotFoundError("Cannot find tree_structures dataset root")


def normalize(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def biased_proba(proba: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return normalize(np.exp(np.log(np.clip(normalize(proba), 1e-12, 1.0)) + bias[None, :]))


def audio_group_keys(audio_files: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    group = audio_files.astype(str).map(lambda x: re.sub(r"_window_\d+.*$", "", Path(x).stem))
    specimen = audio_files.astype(str).map(lambda x: re.sub(r"_segment_.*$", "", Path(x).stem))
    group_codes, _ = pd.factorize(group, sort=False)
    segment_frame = pd.DataFrame({"group": group, "specimen": specimen}).drop_duplicates("group")
    specimen_codes, _ = pd.factorize(segment_frame["specimen"], sort=False)
    return group_codes.astype(np.int64), specimen_codes.astype(np.int64)


def audio_segment_lift(frame: pd.DataFrame, window_proba: np.ndarray) -> np.ndarray:
    group_codes, specimen_codes = audio_group_keys(frame["audio_file"])
    n_groups = int(group_codes.max()) + 1
    logp = np.log(np.clip(normalize(window_proba), 1e-12, 1.0))
    sums = np.vstack([np.bincount(group_codes, weights=logp[:, c], minlength=n_groups) for c in LABELS]).T
    segment = normalize(np.exp(sums - sums.max(axis=1, keepdims=True)))
    output = segment.copy()
    contact_mass = output[:, 1:4].sum(axis=1)
    contact_dist = normalize(output[:, 1:4])
    for specimen_id in np.unique(specimen_codes):
        idx = np.where(specimen_codes == specimen_id)[0]
        contact_idx = idx[contact_mass[idx] >= 0.45]
        if len(contact_idx) < 1:
            continue
        consensus = normalize(np.mean(contact_dist[contact_idx], axis=0, keepdims=True))[0]
        output[idx, 1:4] = contact_mass[idx, None] * consensus[None, :]
        output[idx, 0] = 1.0 - contact_mass[idx]
        if float(np.max(consensus)) < 0.45:
            continue
        pred_before = output[idx].argmax(axis=1)
        lift_idx = idx[(pred_before == 0) & (contact_mass[idx] >= 0.35)]
        if len(lift_idx):
            lifted_mass = np.clip(np.maximum(contact_mass[lift_idx], 0.58), 1e-12, 0.98)
            output[lift_idx, 0] = 1.0 - lifted_mass
            output[lift_idx, 1:4] = lifted_mass[:, None] * consensus[None, :]
    segment_out = normalize(output)
    return segment_out[group_codes]


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    binary_true = (y_true > 0).astype(np.int64)
    binary_pred = (pred > 0).astype(np.int64)
    report = classification_report(
        y_true,
        pred,
        labels=LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    binary_report = classification_report(
        binary_true,
        binary_pred,
        labels=[0, 1],
        target_names=["ambient", "noambient"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "n": int(len(y_true)),
        "accuracy_4class": float(accuracy_score(y_true, pred)),
        "macro_precision_4class": float(precision_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall_4class": float(recall_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1_4class": float(f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1_4class": float(f1_score(y_true, pred, labels=LABELS, average="weighted", zero_division=0)),
        "binary_accuracy_ambient_noambient": float(accuracy_score(binary_true, binary_pred)),
        "binary_macro_precision_ambient_noambient": float(precision_score(binary_true, binary_pred, labels=[0, 1], average="macro", zero_division=0)),
        "binary_macro_recall_ambient_noambient": float(recall_score(binary_true, binary_pred, labels=[0, 1], average="macro", zero_division=0)),
        "binary_macro_f1_ambient_noambient": float(f1_score(binary_true, binary_pred, labels=[0, 1], average="macro", zero_division=0)),
        "per_class_4class": report,
        "per_class_binary": binary_report,
        "confusion_matrix_4class": confusion_matrix(y_true, pred, labels=LABELS).tolist(),
        "confusion_matrix_binary": confusion_matrix(binary_true, binary_pred, labels=[0, 1]).tolist(),
    }


def image_probabilities(root: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dir = root / "audio_visual_dataset_default"
    test_dir = root / "audio_visual_dataset_robo_default"
    train_full = load_image_manifest(train_dir / "dataset.csv", train_dir, "hand_train")
    test = load_image_manifest(test_dir / "dataset.csv", test_dir, "robot_test")
    run = output / "audio_feature_benchmarks" / "image_dl_finetune_resnet18_specimen_select"
    split_dir = run / "splits"
    train_inner = pd.read_csv(split_dir / "hand_train_inner.csv")
    val = pd.read_csv(split_dir / "hand_val.csv")
    checkpoint_path = run / "models" / "image_dl_finetune_resnet18_specimen_select_best_val_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, 4))
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    def predict(frame: pd.DataFrame) -> np.ndarray:
        ds = ImageDataset(frame)
        # Keep evaluation single-process: the managed runtime may forbid the
        # multiprocessing resource-sharer socket used by DataLoader workers.
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        model.eval()
        logits_rows = []
        with torch.inference_mode():
            for images, _ in loader:
                logits_rows.append(model(images.to(device)).detach().cpu().numpy())
        logits = np.concatenate(logits_rows, axis=0)
        return biased_proba(torch.softmax(torch.from_numpy(logits), dim=1).numpy(), IMAGE_BIAS)

    val_out = val.copy()
    val_out[PROBA_COLUMNS] = predict(val)
    test_out = test.copy()
    test_out[PROBA_COLUMNS] = predict(test)
    return train_inner, val_out, test_out


def audio_probabilities(output: Path, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = root / "audio_visual_dataset_default" / "dataset.csv"
    train_raw = pd.read_csv(train_csv)
    train_df = pd.DataFrame({"audio_file": train_raw["audio_file"].astype(str), "y": train_raw["category"].astype(str).str.lower().map({n: i for i, n in enumerate(CLASS_NAMES)}).astype(np.int64)})
    highsr_run = output / "audio_feature_benchmarks" / "audio_highsr_temporal_tta_select"
    lock = json.loads((highsr_run / "reports" / "audio_highsr_temporal_tta_select_selected_without_test.json").read_text())
    candidate = lock["selected_candidate"]
    weights = lock["selected_tta_weights"]
    highsr_oof = normalize(
        float(weights["clean"]) * np.load(highsr_run / "oof_proba" / candidate / "clean_oof_proba.npy")
        + float(weights["robot_mix"]) * np.load(highsr_run / "oof_proba" / candidate / "robot_mix_oof_proba.npy")
    )
    pairwise_oof = normalize(np.load(output / "audio_feature_benchmarks" / "audio_group_consistency_pair_blend_select" / "oof_sources" / "pairwise_selected_clean_oof_proba.npy"))
    audio_oof = audio_segment_lift(train_df, normalize(0.8 * highsr_oof + 0.2 * pairwise_oof))
    oof_out = train_df.copy()
    oof_out[PROBA_COLUMNS] = audio_oof

    test_path = output / "audio_feature_benchmarks" / "audio_lift_source_blend_select" / "reports" / "audio_lift_source_blend_select_final_test_predictions.csv"
    test_out = pd.read_csv(test_path)
    test_out = test_out[["audio_file", "y", *PROBA_COLUMNS]].copy()
    test_out[PROBA_COLUMNS] = normalize(test_out[PROBA_COLUMNS].to_numpy())
    return oof_out, test_out


def main() -> None:
    output = Path("outputs")
    root = resolve_root()
    run = output / "audio_feature_benchmarks" / "image_audio_late_fusion_select_final_test"
    report_dir = run / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    print("ROOT_PATH =", root.resolve(), flush=True)
    print("Protocol  = image/audio late fusion; alpha selected on hand_val only; robot/test after lock", flush=True)

    _, image_val, image_test = image_probabilities(root, output)
    audio_oof, audio_test = audio_probabilities(output, root)

    val = image_val[["audio_file", "y", *PROBA_COLUMNS]].merge(
        audio_oof[["audio_file", "y", *PROBA_COLUMNS]], on="audio_file", suffixes=("_image", "_audio"), validate="one_to_one"
    )
    if not np.array_equal(val["y_image"].to_numpy(), val["y_audio"].to_numpy()):
        raise AssertionError("Image/audio validation labels are not aligned")
    y_val = val["y_image"].to_numpy(dtype=np.int64)
    p_img_val = val[[f"{c}_image" for c in PROBA_COLUMNS]].to_numpy()
    p_audio_val = val[[f"{c}_audio" for c in PROBA_COLUMNS]].to_numpy()

    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        p = normalize(alpha * p_audio_val + (1.0 - alpha) * p_img_val)
        pred = p.argmax(axis=1)
        m = metrics(y_val, pred)
        candidates.append({"audio_weight": float(alpha), "image_weight": float(1.0 - alpha), **m})
    leaderboard = pd.DataFrame([
        {k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in candidates
    ]).sort_values(["macro_f1_4class", "binary_macro_f1_ambient_noambient", "accuracy_4class"], ascending=False).reset_index(drop=True)
    selected = candidates[int(leaderboard.index[0])]
    # leaderboard index was reset; recover the selected row explicitly.
    selected = next(row for row in candidates if row["audio_weight"] == float(leaderboard.iloc[0]["audio_weight"]))
    json_lock = {
        "protocol": "image_audio_late_fusion_no_test_tuning",
        "selection_split": "hand_val",
        "forbidden_selection_data": "robot/test labels and robot/test probabilities",
        "image_model": "existing locked ResNet18 hand-train checkpoint with existing hand-val class bias",
        "audio_model": "existing locked audio_lift_source_blend anchor + report_gate_onehot + segment_lift",
        "selected_audio_weight": selected["audio_weight"],
        "selected_image_weight": selected["image_weight"],
        "selection_metrics": {k: v for k, v in selected.items() if k not in {"per_class_4class", "per_class_binary", "confusion_matrix_4class", "confusion_matrix_binary"}},
    }
    (report_dir / "image_audio_late_fusion_select_selected_without_test.json").write_text(json.dumps(json_lock, indent=2, default=float), encoding="utf-8")
    leaderboard.to_csv(report_dir / "image_audio_late_fusion_select_hand_val_leaderboard.csv", index=False)
    print("Selection lock written before robot/test:", json.dumps(json_lock, indent=2, default=float), flush=True)

    test = image_test[["audio_file", "y", *PROBA_COLUMNS]].merge(
        audio_test[["audio_file", "y", *PROBA_COLUMNS]], on="audio_file", suffixes=("_image", "_audio"), validate="one_to_one"
    )
    if not np.array_equal(test["y_image"].to_numpy(), test["y_audio"].to_numpy()):
        raise AssertionError("Image/audio test labels are not aligned")
    y_test = test["y_image"].to_numpy(dtype=np.int64)
    p_img_test = test[[f"{c}_image" for c in PROBA_COLUMNS]].to_numpy()
    p_audio_test = test[[f"{c}_audio" for c in PROBA_COLUMNS]].to_numpy()
    p_fused = normalize(selected["audio_weight"] * p_audio_test + selected["image_weight"] * p_img_test)
    pred_test = p_fused.argmax(axis=1).astype(np.int64)
    final_metrics = metrics(y_test, pred_test)
    final_metrics.update({"split": "robot_test_final", "audio_weight": selected["audio_weight"], "image_weight": selected["image_weight"]})
    (report_dir / "image_audio_late_fusion_select_final_test_metrics.json").write_text(json.dumps(final_metrics, indent=2, default=float), encoding="utf-8")
    pd.DataFrame([{
        k: v for k, v in final_metrics.items() if not isinstance(v, (dict, list))
    }]).to_csv(report_dir / "image_audio_late_fusion_select_final_test_report.csv", index=False)
    pd.DataFrame(final_metrics["confusion_matrix_4class"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(report_dir / "image_audio_late_fusion_select_final_test_confusion_matrix_4class.csv")
    pd.DataFrame(final_metrics["confusion_matrix_binary"], index=["ambient", "noambient"], columns=["ambient", "noambient"]).to_csv(report_dir / "image_audio_late_fusion_select_final_test_confusion_matrix_binary.csv")
    pred_frame = test[["audio_file", "y_image"]].rename(columns={"y_image": "y"}).copy()
    pred_frame["pred_y"] = pred_test
    pred_frame["pred_label"] = [CLASS_NAMES[i] for i in pred_test]
    pred_frame["audio_weight"] = selected["audio_weight"]
    pred_frame["image_weight"] = selected["image_weight"]
    for index, name in enumerate(CLASS_NAMES):
        pred_frame[f"proba_{name}"] = p_fused[:, index]
    pred_frame.to_csv(report_dir / "image_audio_late_fusion_select_final_test_predictions.csv", index=False)
    print("Final robot/test metrics:", json.dumps(final_metrics, indent=2, default=float), flush=True)
    print("Reports:", report_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
