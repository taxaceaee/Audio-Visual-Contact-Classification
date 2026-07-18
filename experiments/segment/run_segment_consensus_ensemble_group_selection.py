"""Segment-pure ensemble + contact-aware material consensus lift.

Builds on hierarchical OOF arrays. Hand-only selection; never loads robot/test.
"""
from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_avr_group_selection as avr
import run_multimodal_val_locked_suite as suite
import train_audio_broad_oof_meta_select_final_test as broad
import train_audio_group_consistency_pair_blend_select_final_test as group_audio
import train_audio_lift_source_blend_select_final_test as lift
import train_audio_specimen_contact_consensus_select_final_test as specimen
import train_val_select_final_test as audio_base

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path(
    os.environ.get(
        "SEG_OUT",
        "outputs/audio_feature_benchmarks/segment_consensus_ensemble_group_selection",
    )
)
IMG_CLIP = Path(
    "outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k"
)
IMG_EFF = Path("outputs/image_timm_features/efficientnet_b3.ra2_in1k")
IMG_R18 = Path("outputs/image_deep_features/resnet18_224")


def seg(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"_window_\d+.*$", "", regex=True)


def spec(s: pd.Series) -> pd.Series:
    return seg(s).str.replace(r"_segment_.*$", "", regex=True)


def fast_macro(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for c in range(4):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def oof_mat(x, y, folds, C=0.03):
    oof = np.zeros((len(y), 3), np.float32)
    for tr, va in folds:
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C, max_iter=1500, class_weight="balanced", multi_class="multinomial", random_state=42
            ),
        )
        ci = tr[y[tr] > 0]
        m.fit(x[ci], y[ci] - 1)
        oof[va] = m.predict_proba(x[va])
    return oof


def oof_bin(x, y, folds, C=0.03):
    oof = np.zeros(len(y), np.float32)
    for tr, va in folds:
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=1500, class_weight="balanced", random_state=42),
        )
        m.fit(x[tr], (y[tr] > 0).astype(int))
        oof[va] = m.predict_proba(x[va])[:, 1]
    return oof


def oof_4(x, y, folds, C=0.03):
    oof = np.zeros((len(y), 4), np.float32)
    for tr, va in folds:
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C, max_iter=1500, class_weight="balanced", multi_class="multinomial", random_state=42
            ),
        )
        m.fit(x[tr], y[tr])
        oof[va] = suite.normalize(m.predict_proba(x[va]))
    return oof


def consensus_lift_material(
    mat: np.ndarray,
    contact: np.ndarray,
    specimen_ids: np.ndarray,
    contact_th: float,
    min_contact_segments: int = 1,
    consensus_strength: float = 0.5,
) -> np.ndarray:
    """Average material among contact segments of same specimen; blend back.

    Does not use specimen labels — only prediction-side consensus (valid).
    """
    out = mat.copy()
    for sid in np.unique(specimen_ids):
        idx = np.where(specimen_ids == sid)[0]
        cidx = idx[contact[idx] >= contact_th]
        if len(cidx) < min_contact_segments:
            continue
        cons = suite.normalize(mat[cidx].mean(0, keepdims=True))[0]
        out[cidx] = suite.normalize(
            (1.0 - consensus_strength) * mat[cidx] + consensus_strength * cons[None, :]
        )
    return out


def predict_proba4(contact, mat, th):
    """Build 4-class probs: ambient mass from gate, material fills rest."""
    c = np.clip(contact, 0, 1)
    # soft gate: use threshold as hard but store soft for ensemble
    ambient = np.where(c >= th, 1.0 - c, 1.0 - np.minimum(c, th * 0.5))
    # better hard construction for argmax:
    p = np.zeros((len(contact), 4), dtype=np.float64)
    contact_mass = np.where(c >= th, np.maximum(c, 0.55), 0.0)
    p[:, 0] = 1.0 - contact_mass
    p[:, 1:] = contact_mass[:, None] * mat
    # for non-contact force ambient
    p[c < th] = np.array([1.0, 0.0, 0.0, 0.0])
    return suite.normalize(p)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audio_base.configure_feature_set("total240")
    fr = suite.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv",
        ROOT / "audio_visual_dataset_default",
        "hand_train",
    )
    y = fr.y.to_numpy(np.int64)
    files = fr.audio_file.astype(str)
    sk = seg(files).to_numpy()
    sg = spec(files).to_numpy()
    useg, codes = np.unique(sk, return_inverse=True)
    sy = np.array([y[codes == i][0] for i in range(len(useg))])
    ss = np.array([sg[codes == i][0] for i in range(len(useg))])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(len(sy)), sy, ss
        )
    )

    af = audio_base.load_manifest(
        ROOT / "audio_visual_dataset_default/dataset.csv", "hand_train"
    )
    high = group_audio.load_highsr_oof(Path("outputs"))
    pair = specimen.load_pairwise_oof(
        Path(
            "outputs/audio_feature_benchmarks/audio_log_consensus_pair_blend_select/"
            "oof_sources/pairwise_selected_clean_oof_proba.npy"
        )
    )
    anchor = lift.anchor_lift_proba(af, high, pair)
    assignment = np.full(len(y), -1, np.int64)
    for k, (_, va) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.arange(len(y)), y, sg
        )
    ):
        assignment[va] = k
    src = broad.load_train_sources(Path("outputs"), y, assignment, 42)
    audio = lift.postprocess(
        af,
        lift.normalize(0.95 * anchor + 0.05 * src["report_gate_onehot"]),
        "segment_lift",
    )
    pc, lg = avr.avr_inputs(audio)
    seg_pc = np.array([pc[codes == i].mean() for i in range(len(useg))])
    seg_audio_mat = np.vstack(
        [
            suite.normalize(np.exp(lg[codes == i].mean(0, keepdims=True)))[0]
            for i in range(len(useg))
        ]
    )
    # also full audio4
    seg_audio4 = np.vstack(
        [suite.normalize(audio[codes == i].mean(0, keepdims=True))[0] for i in range(len(useg))]
    )

    print("building multi-backbone OOF...", flush=True)
    xs = {}
    for name, path in [
        ("clip", IMG_CLIP),
        ("eff", IMG_EFF),
        ("r18", IMG_R18),
    ]:
        x = np.load(path / "hand_train_full/X.npy").astype(np.float32)
        xs[name] = np.vstack([x[codes == i].mean(0) for i in range(len(useg))])
    xs["cat"] = np.hstack([xs["clip"], xs["eff"]])
    xs["cat3"] = np.hstack([xs["clip"], xs["eff"], xs["r18"]])

    mat_oof = {k: oof_mat(v, sy, folds) for k, v in xs.items()}
    bin_oof = {k: oof_bin(xs[k], sy, folds) for k in ("clip", "cat", "cat3")}
    four_oof = {k: oof_4(xs[k], sy, folds) for k in ("clip", "cat")}
    trunk_oof = np.zeros(len(sy), np.float32)
    for tr, va in folds:
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=1200, class_weight="balanced", random_state=42),
        )
        m.fit(xs["clip"][tr], (sy[tr] == 2).astype(int))
        trunk_oof[va] = m.predict_proba(xs["clip"][va])[:, 1]

    np.savez_compressed(
        OUT / "hand_oof_cache.npz",
        sy=sy,
        ss=ss,
        seg_pc=seg_pc,
        seg_audio_mat=seg_audio_mat,
        seg_audio4=seg_audio4,
        trunk_oof=trunk_oof,
        **{f"mat_{k}": v for k, v in mat_oof.items()},
        **{f"bin_{k}": v for k, v in bin_oof.items()},
        **{f"four_{k}": v for k, v in four_oof.items()},
    )

    # Generate diverse candidate 4-class OOF probability matrices
    candidates = []
    ths = [0.40, 0.48, 0.55, 0.60, 0.65]
    for csrc, alpha in [("audio", 1.0), ("clip", 0.5), ("clip", 0.7), ("cat", 0.5), ("cat3", 0.6)]:
        if csrc == "audio":
            contact = seg_pc
        else:
            contact = alpha * seg_pc + (1 - alpha) * bin_oof[csrc]
        for msrc, wl, wt, ww in product(
            ["clip", "eff", "cat", "cat3", "audio"],
            [0.3, 0.5, 0.7],
            [0.4, 0.55, 0.7],
            [0.3, 0.5, 0.7],
        ):
            if msrc == "audio":
                mat = seg_audio_mat
            else:
                w = np.array([wl, wt, ww], dtype=np.float64)
                mat = suite.normalize(seg_audio_mat * (1 - w) + mat_oof[msrc] * w)
            for tb in (0.0, 0.3, 0.5):
                if tb:
                    logits = np.log(np.clip(mat, 1e-8, 1))
                    logits[:, 1] += tb * trunk_oof
                    mat_b = suite.normalize(np.exp(logits))
                else:
                    mat_b = mat
                for th in ths:
                    for cstr in (0.0, 0.4, 0.7):
                        mat_c = (
                            consensus_lift_material(mat_b, contact, ss, th, 1, cstr)
                            if cstr > 0
                            else mat_b
                        )
                        p4 = predict_proba4(contact, mat_c, th)
                        pred = p4.argmax(1)
                        f1 = fast_macro(sy, pred)
                        candidates.append(
                            {
                                "contact_source": csrc,
                                "contact_alpha_audio": float(alpha),
                                "material_source": msrc,
                                "leaf_w": float(wl),
                                "trunk_w": float(wt),
                                "twig_w": float(ww),
                                "trunk_bias": float(tb),
                                "contact_threshold": float(th),
                                "consensus_strength": float(cstr),
                                "macro_f1_4class": float(f1),
                                "binary_macro_f1": float(
                                    f1_score(sy > 0, pred > 0, average="macro", zero_division=0)
                                ),
                                "p4": p4,
                            }
                        )
    # sort unique-ish by score
    candidates.sort(key=lambda r: (r["macro_f1_4class"], r["binary_macro_f1"]), reverse=True)
    # keep top unique configs
    seen = set()
    top = []
    for c in candidates:
        key = (
            c["contact_source"],
            c["contact_alpha_audio"],
            c["material_source"],
            c["leaf_w"],
            c["trunk_w"],
            c["twig_w"],
            c["trunk_bias"],
            c["contact_threshold"],
            c["consensus_strength"],
        )
        if key in seen:
            continue
        seen.add(key)
        top.append(c)
        if len(top) >= 40:
            break

    # Ensemble: average top-k OOF probs, select k and optional temp on hand
    ens_rows = []
    for k in (1, 3, 5, 7, 10, 15):
        for temp in (0.8, 1.0, 1.2):
            stack = np.mean([t["p4"] ** (1.0 / temp) for t in top[:k]], axis=0)
            stack = suite.normalize(stack)
            pred = stack.argmax(1)
            ens_rows.append(
                {
                    "kind": "ensemble",
                    "k": k,
                    "temperature": temp,
                    "macro_f1_4class": fast_macro(sy, pred),
                    "binary_macro_f1": float(
                        f1_score(sy > 0, pred > 0, average="macro", zero_division=0)
                    ),
                    "trunk_recall": float(
                        np.sum((sy == 2) & (pred == 2)) / max(np.sum(sy == 2), 1)
                    ),
                }
            )
    # single best
    best_single = {
        "kind": "single",
        "k": 1,
        "temperature": 1.0,
        **{kk: top[0][kk] for kk in top[0] if kk != "p4"},
    }
    board_ens = (
        pd.DataFrame(ens_rows)
        .sort_values(["macro_f1_4class", "binary_macro_f1"], ascending=False)
        .reset_index(drop=True)
    )
    board_single = pd.DataFrame([{kk: top[i][kk] for kk in top[i] if kk != "p4"} for i in range(min(20, len(top)))])
    board_single.to_csv(OUT / "hand_single_leaderboard.csv", index=False)
    board_ens.to_csv(OUT / "hand_ensemble_leaderboard.csv", index=False)

    # pick best among ensemble and single by hand macro f1
    best_ens = board_ens.iloc[0].to_dict()
    if best_ens["macro_f1_4class"] >= best_single["macro_f1_4class"]:
        selected = {
            "kind": "ensemble",
            "k": int(best_ens["k"]),
            "temperature": float(best_ens["temperature"]),
            "mean_cv_macro_f1": float(best_ens["macro_f1_4class"]),
            "members": [
                {kk: top[i][kk] for kk in top[i] if kk != "p4"}
                for i in range(int(best_ens["k"]))
            ],
        }
    else:
        selected = {
            "kind": "single",
            "k": 1,
            "temperature": 1.0,
            "mean_cv_macro_f1": float(best_single["macro_f1_4class"]),
            "members": [{kk: top[0][kk] for kk in top[0] if kk != "p4"}],
        }

    # save top members config for final
    lock = {
        "protocol": "segment_consensus_ensemble_material_group_OOF",
        "selection_data": "hand/default only",
        "group_column": "specimen_group",
        "audio_source": "locked group-aware OOF audio",
        "image_source": "CLIP+Eff+ResNet18 segment OOF material/contact",
        "test_loaded": False,
        "selected_candidate": selected,
        "n_single_candidates_scored": int(len(candidates)),
    }
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2, default=float))
    print(json.dumps(lock, indent=2, default=float))
    print("top ensemble:\n", board_ens.head(8).to_string(index=False))
    print("top single:\n", board_single.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
