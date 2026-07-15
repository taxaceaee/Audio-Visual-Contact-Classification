"""Pure-paper gt09 core: multi-backbone material soft on frozen contact stack.

Works from feature caches (paths/y/X) when raw dataset root is offline.
Selection uses hand only; robot features are for final-test apply after lock.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]

# Only backbones with both hand_train_full and robot_test caches.
IMG = {
    "clip": REPO / "outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k",
    "eff": REPO / "outputs/image_timm_features/efficientnet_b3.ra2_in1k",
    "dino": REPO / "outputs/image_timm_features/vit_small_patch14_dinov2.lvd142m",
    "convnext": REPO / "outputs/image_timm_features/convnext_tiny.fb_in22k_ft_in1k",
}
AUDIO_CLEAN_H = REPO / "outputs/audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/X.npy"
AUDIO_CLEAN_R = REPO / "outputs/audio_feature_benchmarks/total240_trainval_select/features/robot_test/X.npy"
AUDIO_MIX_H = REPO / "outputs/audio_feature_benchmarks/total240_stress_cv_select/stress_features/robot_mix/X.npy"
W2V_H = REPO / "outputs/audio_wav2vec2_features/hand_train_full_X.npy"
W2V_R = REPO / "outputs/audio_wav2vec2_features/robot_test_X.npy"
HAND_PATHS = REPO / "outputs/audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/paths.npy"
HAND_Y = REPO / "outputs/audio_feature_benchmarks/total240_trainval_select/features/hand_train_full/y.npy"
ROBOT_PATHS = REPO / "outputs/audio_feature_benchmarks/total240_trainval_select/features/robot_test/paths.npy"
ROBOT_Y = REPO / "outputs/audio_feature_benchmarks/total240_trainval_select/features/robot_test/y.npy"
V2_BASE = REPO / (
    "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/"
    "segment_rule_stack_v2_base_segment_outputs.npz"
)
HAND_BUNDLE = REPO / (
    "outputs/audio_feature_benchmarks/multimodal_085_material_logit_bias_group_selection/"
    "hand_oof_bundle.npz"
)


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def segment_key(path: str) -> str:
    """Segment id aligned with v2 base (`audio/<stem_without_window>`)."""
    p = str(path).replace("\\", "/")
    name = Path(p).name  # file name
    stem = Path(name).stem
    stem = re.sub(r"_window_\d+.*$", "", stem)
    # v2 NPZ stores keys as audio/<segment>
    return f"audio/{stem}"


def specimen_key(path: str) -> str:
    stem = Path(str(path)).stem
    stem = re.sub(r"_window_\d+.*$", "", stem)
    return re.sub(r"_segment_.*$", "", stem)


def pool_segment_from_paths(paths: np.ndarray, x: np.ndarray, y: np.ndarray | None = None):
    """Pool windows→segments with np.unique order (matches hand_oof_bundle)."""
    keys = np.array([segment_key(p) for p in paths.astype(str)])
    u, inv = np.unique(keys, return_inverse=True)
    px = np.vstack([x[inv == i].mean(0) for i in range(len(u))]).astype(np.float32)
    py = None if y is None else np.array([y[inv == i][0] for i in range(len(u))], np.int64)
    specs = np.array([specimen_key(paths.astype(str)[inv == i][0]) for i in range(len(u))])
    return u, inv, px, py, specs


def load_hand_window():
    paths = np.load(HAND_PATHS, allow_pickle=True)
    y = np.load(HAND_Y).astype(np.int64)
    return paths, y


def load_robot_window():
    paths = np.load(ROBOT_PATHS, allow_pickle=True)
    y = np.load(ROBOT_Y).astype(np.int64)
    return paths, y


def load_multi_bb(split: str) -> np.ndarray:
    """split: hand_train_full | robot_test"""
    xs = []
    for name, root in IMG.items():
        p = root / split / "X.npy"
        xs.append(np.load(p).astype(np.float32))
    return np.hstack(xs)


def apply_contact(pred_v2, oof_ts, oof_cs, oof_bin, oof_tr, primary, secondary):
    out = pred_v2.copy().astype(np.int64)
    out[
        (out == 0)
        & (oof_ts >= primary["ts_th"])
        & (np.maximum(oof_cs, oof_bin) >= primary["cs_th"])
    ] = 2
    out[
        (out == 0)
        & (oof_tr >= secondary["img_th"])
        & (oof_bin >= secondary["img_cs"])
    ] = 2
    return out


def cascade_contaminate(soft, y, a_tt, a_tl):
    """Hand-only surrogate material contaminants for selection (no robot)."""
    out = soft.copy()
    m = y == 2
    if m.any() and a_tt > 0:
        out[m, 3] = out[m, 3] + a_tt * out[m, 2]
        out[m, 2] = (1.0 - a_tt) * out[m, 2]
        out[m] = normalize(out[m])
    m = y == 3
    if m.any() and a_tl > 0:
        out[m, 1] = out[m, 1] + a_tl * out[m, 3]
        out[m, 3] = (1.0 - a_tl) * out[m, 3]
        out[m] = normalize(out[m])
    return out


def decode_material(contact, soft, b_leaf=0.0, b_trunk=0.0, b_twig=0.0, T=1.0):
    """Protect-trunk: redecode only leaf/twig preds."""
    out = contact.copy().astype(np.int64)
    m = (out == 1) | (out == 3)
    if not m.any():
        return out
    logits = np.log(np.clip(soft[m][:, 1:4], 1e-12, None)) / float(T)
    logits = logits + np.array([float(b_leaf), float(b_trunk), float(b_twig)], dtype=np.float64)
    out[m] = 1 + logits.argmax(1)
    return out


def blend_soft(meta, multibb, alpha: float) -> np.ndarray:
    a = float(alpha)
    return normalize((1.0 - a) * meta + a * multibb)


def macro_f1(y, pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, pred, average="macro", zero_division=0))


def build_hand_multibb_oof(n_splits=5, random_state=42, C=0.05):
    """Group-OOF 4-class probs from multi-bb image ‖ total240 ‖ wav2vec2 at segment level."""
    paths, y = load_hand_window()
    img = load_multi_bb("hand_train_full")
    audio = np.load(AUDIO_CLEAN_H).astype(np.float32)
    w2v = np.load(W2V_H).astype(np.float32)
    xw = np.hstack([img, audio, w2v])
    u, inv, xs, ys, specs = pool_segment_from_paths(paths, xw, y)

    bundle = np.load(HAND_BUNDLE)
    assert len(bundle["sy"]) == len(ys)
    assert np.array_equal(bundle["sy"], ys), "segment order must match hand_oof_bundle"

    oof = np.zeros((len(ys), 4), dtype=np.float64)
    sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=random_state)
    for tr, va in sgkf.split(np.arange(len(ys)), ys, specs):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=4000,
                class_weight="balanced",
                multi_class="multinomial",
                random_state=random_state,
            ),
        )
        clf.fit(xs[tr], ys[tr])
        oof[va] = clf.predict_proba(xs[va])
    return {
        "segment_ids": u,
        "sy": ys,
        "specs": specs,
        "oof_multibb4": normalize(oof),
        "X_seg": xs,
        "inv_window": inv,
        "y_window": y,
        "paths": paths,
    }


def fit_hand_multibb_full(C=0.05, random_state=42):
    """Fit multi-bb material model on all hand segments (for robot apply)."""
    paths, y = load_hand_window()
    img = load_multi_bb("hand_train_full")
    audio = np.load(AUDIO_CLEAN_H).astype(np.float32)
    w2v = np.load(W2V_H).astype(np.float32)
    xw = np.hstack([img, audio, w2v])
    u, inv, xs, ys, specs = pool_segment_from_paths(paths, xw, y)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=4000,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=random_state,
        ),
    )
    clf.fit(xs, ys)
    return clf, u, ys, specs


def robot_multibb_proba(clf) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths, y = load_robot_window()
    img = load_multi_bb("robot_test")
    audio = np.load(AUDIO_CLEAN_R).astype(np.float32)
    w2v = np.load(W2V_R).astype(np.float32)
    xw = np.hstack([img, audio, w2v])
    u, inv, xs, ys, specs = pool_segment_from_paths(paths, xw, y)
    proba = normalize(clf.predict_proba(xs))
    return u, inv, proba, y


def expand_segment_pred(seg_ids, pred_seg, window_paths) -> np.ndarray:
    order = {str(k): i for i, k in enumerate(seg_ids)}
    keys = [segment_key(p) for p in window_paths.astype(str)]
    return np.array([pred_seg[order[k]] for k in keys], dtype=np.int64)


def robot_contact_and_meta(primary, secondary):
    """Fit detectors hand-only; apply contact on robot v2 base. Returns seg arrays."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # hand fit detectors (clear-style: clip for bin/tr; mix‖w2v‖clip for ts/cs)
    h_paths, h_y = load_hand_window()
    clip_h = np.load(IMG["clip"] / "hand_train_full" / "X.npy").astype(np.float32)
    hu, hinv, clip_s, hy, _ = pool_segment_from_paths(h_paths, clip_h, h_y)
    mix = np.load(AUDIO_MIX_H).astype(np.float32)
    mix_s = np.vstack([mix[hinv == i].mean(0) for i in range(len(hu))])
    w2v_h = np.load(W2V_H).astype(np.float32)
    w2v_s = np.vstack([w2v_h[hinv == i].mean(0) for i in range(len(hu))])
    # multi-bb for stronger amb det (claim v2 full path style uses multi image)
    multi_h = []
    for name in IMG:
        x = np.load(IMG[name] / "hand_train_full" / "X.npy").astype(np.float32)
        multi_h.append(np.vstack([x[hinv == i].mean(0) for i in range(len(hu))]))
    multi_h = np.hstack(multi_h)
    X_amb = np.hstack([mix_s, w2v_s, multi_h])

    def fit_bin(x, ybin, C=0.05):
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=42),
        )
        m.fit(x, ybin)
        return m

    def pos(m, x):
        p = m.predict_proba(x)
        cl = list(m.classes_)
        return p[:, cl.index(1)] if 1 in cl else p[:, -1]

    m_bin = fit_bin(clip_s, (hy > 0).astype(int), 0.03)
    m_tr = fit_bin(clip_s, (hy == 2).astype(int), 0.1)
    m_ts = fit_bin(X_amb, (hy == 2).astype(int), 0.05)
    m_cs = fit_bin(X_amb, (hy > 0).astype(int), 0.05)

    z = np.load(V2_BASE, allow_pickle=True)
    ids = z["segment_ids"].astype(str)
    pred_v2 = np.where(z["pred_h"] == 2, z["pred_h"], z["pred_m"]).astype(np.int64)
    p_meta = z["p_meta"].astype(np.float64)

    r_paths, r_y = load_robot_window()
    clip_r = np.load(IMG["clip"] / "robot_test" / "X.npy").astype(np.float32)
    tu, tinv, clip_t, _, _ = pool_segment_from_paths(r_paths, clip_r, r_y)
    multi_t = []
    for name in IMG:
        x = np.load(IMG[name] / "robot_test" / "X.npy").astype(np.float32)
        multi_t.append(np.vstack([x[tinv == i].mean(0) for i in range(len(tu))]))
    multi_t = np.hstack(multi_t)
    ta = np.load(AUDIO_CLEAN_R).astype(np.float32)
    ta_s = np.vstack([ta[tinv == i].mean(0) for i in range(len(tu))])
    tw = np.load(W2V_R).astype(np.float32)
    tw_s = np.vstack([tw[tinv == i].mean(0) for i in range(len(tu))])
    Xr = np.hstack([ta_s, tw_s, multi_t])

    order = {k: i for i, k in enumerate(ids)}
    pred = np.array([pred_v2[order[str(k)]] for k in tu], dtype=np.int64)
    soft_meta = np.array([p_meta[order[str(k)]] for k in tu], dtype=np.float64)

    bins = pos(m_bin, clip_t)
    tr = pos(m_tr, clip_t)
    ts = pos(m_ts, Xr)
    cs = pos(m_cs, Xr)
    contact = apply_contact(pred, ts, cs, bins, tr, primary, secondary)
    return {
        "segment_ids": tu,
        "contact": contact,
        "soft_meta": soft_meta,
        "y_window": r_y,
        "paths": r_paths,
        "inv": tinv,
    }
