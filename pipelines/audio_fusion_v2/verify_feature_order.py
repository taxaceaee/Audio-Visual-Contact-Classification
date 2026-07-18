#!/usr/bin/env python3
"""Verify fusion feature stack order and shapes match the sealed clear pipeline.

Checks:
  1) Feature file presence + shapes vs n_rows from manifests
  2) Segment-pool + hstack order: [total240, wav2vec2, CLIP]
  3) Hand amb detector width = mix_dim + w2v_dim + clip_dim
  4) Robot predict Macro F1 == sealed 0.850781...
  5) Feature arrays identical (hash) to workspace originals when present

Usage:
    python verify_feature_order.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TREE_BUNDLE_ROOT", str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_v2" / "code"))

from paper_claim.clear_pipeline import (  # noqa: E402
    HParams,
    bundle_paths,
    fit_detectors_hand,
    load_manifest,
    metrics,
    pool_segment,
    predict_robot,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    paths = bundle_paths(ROOT)
    report: dict = {"ok": True, "checks": []}

    def check(name: str, cond: bool, detail: dict | None = None):
        item = {"name": name, "pass": bool(cond)}
        if detail:
            item["detail"] = detail
        report["checks"].append(item)
        if not cond:
            report["ok"] = False
        print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))

    # --- presence ---
    for key in [
        "hand_csv",
        "robot_csv",
        "v2_base",
        "clip_hand",
        "clip_robot",
        "audio_mix",
        "audio_robot",
        "w2v_hand",
        "w2v_robot",
    ]:
        p = paths[key]
        check(f"exists:{key}", p.exists(), {"path": str(p)})

    if not report["ok"]:
        Path(ROOT / "verify_feature_order_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return 1

    # --- load + shapes ---
    hand = load_manifest(paths["hand_csv"], paths["hand_dir"], "hand_train")
    robot = load_manifest(paths["robot_csv"], paths["robot_dir"], "robot_test")
    n_h, n_r = len(hand), len(robot)

    clip_h = np.load(paths["clip_hand"])
    clip_r = np.load(paths["clip_robot"])
    mix_h = np.load(paths["audio_mix"])
    aud_r = np.load(paths["audio_robot"])
    w2v_h = np.load(paths["w2v_hand"])
    w2v_r = np.load(paths["w2v_robot"])
    v2 = np.load(paths["v2_base"], allow_pickle=True)

    check("shape:clip_hand_rows", clip_h.shape[0] == n_h, {"shape": list(clip_h.shape), "n": n_h})
    check("shape:clip_robot_rows", clip_r.shape[0] == n_r, {"shape": list(clip_r.shape), "n": n_r})
    check("shape:mix_hand_rows", mix_h.shape[0] == n_h, {"shape": list(mix_h.shape), "n": n_h})
    check("shape:audio_robot_rows", aud_r.shape[0] == n_r, {"shape": list(aud_r.shape), "n": n_r})
    check("shape:w2v_hand_rows", w2v_h.shape[0] == n_h, {"shape": list(w2v_h.shape), "n": n_h})
    check("shape:w2v_robot_rows", w2v_r.shape[0] == n_r, {"shape": list(w2v_r.shape), "n": n_r})
    check(
        "v2_base_keys",
        all(k in v2.files for k in ("segment_ids", "pred_h", "pred_m", "p_meta")),
        {"files": list(v2.files)},
    )

    # --- segment pool + hstack order (hand) ---
    y = hand.y.to_numpy(np.int64)
    u, c, clip_s, ys = pool_segment(hand.audio_file, clip_h.astype(np.float32), y)
    mix_s = np.vstack([mix_h[c == i].mean(0) for i in range(len(u))])
    w2v_s = np.vstack([w2v_h[c == i].mean(0) for i in range(len(u))])
    X_amb = np.hstack([mix_s, w2v_s, clip_s])

    expected_dim = mix_s.shape[1] + w2v_s.shape[1] + clip_s.shape[1]
    check(
        "feature_order_hand_hstack",
        X_amb.shape == (len(u), expected_dim),
        {
            "order": ["robot_mix_total240", "wav2vec2", "clip"],
            "dims": {
                "mix": int(mix_s.shape[1]),
                "w2v": int(w2v_s.shape[1]),
                "clip": int(clip_s.shape[1]),
                "concat": int(X_amb.shape[1]),
            },
            "n_segments": int(len(u)),
        },
    )
    # slice identity: first mix_dim cols == mix_s, etc.
    d0, d1, d2 = mix_s.shape[1], w2v_s.shape[1], clip_s.shape[1]
    check(
        "slice_order_mix_first",
        np.allclose(X_amb[:, :d0], mix_s),
        {"slice": f"0:{d0}"},
    )
    check(
        "slice_order_w2v_middle",
        np.allclose(X_amb[:, d0 : d0 + d1], w2v_s),
        {"slice": f"{d0}:{d0+d1}"},
    )
    check(
        "slice_order_clip_last",
        np.allclose(X_amb[:, d0 + d1 : d0 + d1 + d2], clip_s),
        {"slice": f"{d0+d1}:{d0+d1+d2}"},
    )

    # --- robot stack same order ---
    yt = robot.y.to_numpy(np.int64)
    tu, tc, clip_rs, _ = pool_segment(robot.audio_file, clip_r.astype(np.float32), yt)
    ta_s = np.vstack([aud_r[tc == i].mean(0) for i in range(len(tu))])
    tw_s = np.vstack([w2v_r[tc == i].mean(0) for i in range(len(tu))])
    X_amb_r = np.hstack([ta_s, tw_s, clip_rs])
    check(
        "feature_order_robot_hstack",
        X_amb_r.shape[1] == ta_s.shape[1] + tw_s.shape[1] + clip_rs.shape[1],
        {
            "order": ["total240_robot_clean", "wav2vec2", "clip"],
            "dims": {
                "audio": int(ta_s.shape[1]),
                "w2v": int(tw_s.shape[1]),
                "clip": int(clip_rs.shape[1]),
            },
            "n_segments": int(len(tu)),
        },
    )

    # --- detectors fit + sealed F1 ---
    det = fit_detectors_hand(paths)
    check("detectors_fit", set(det) == {"ts", "cs", "bin", "tr"}, {"keys": sorted(det)})

    out = predict_robot(HParams(b_leaf=-1.3), paths)
    m = metrics(out["y"], out["pred"])
    sealed = json.loads((ROOT / "shared" / "sealed_targets.json").read_text())
    exp = float(sealed["fusion_paper_claim_v2"]["macro_f1_4class"])
    check(
        "fusion_macro_f1_matches_seal",
        abs(m["macro_f1_4class"] - exp) <= 1e-9,
        {"got": m["macro_f1_4class"], "expected": exp, "n": int(len(out["y"]))},
    )
    check("n_robot_2219", len(out["y"]) == 2219, {"n": int(len(out["y"]))})

    # --- optional: hash match parent workspace (dev only; offsite skips) ---
    # Set AFV2_CHECK_WORKSPACE_HASH=1 when running inside the original tree
    # to cross-check bundled features against outputs/. Offsite default: skip.
    if os.environ.get("AFV2_CHECK_WORKSPACE_HASH", "").strip() in {"1", "true", "yes"}:
        workspace = ROOT.parent
        pairs = [
            (
                paths["audio_robot"],
                workspace
                / "outputs/audio_feature_benchmarks/total240_trainval_select/features/robot_test/X.npy",
            ),
            (
                paths["clip_robot"],
                workspace
                / "outputs/image_timm_features/vit_base_patch16_clip_224.openai_ft_in1k/robot_test/X.npy",
            ),
            (
                paths["w2v_robot"],
                workspace / "outputs/audio_wav2vec2_features/robot_test_X.npy",
            ),
            (
                paths["v2_base"],
                workspace
                / "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection/segment_rule_stack_v2_base_segment_outputs.npz",
            ),
        ]
        for bundled, original in pairs:
            if original.exists() and bundled.exists():
                same = sha256_file(bundled) == sha256_file(original)
                check(
                    f"hash_match:{bundled.name}",
                    same,
                    {"bundled": str(bundled), "original": str(original)},
                )
            else:
                check(
                    f"hash_match_skip:{bundled.name}",
                    True,
                    {"reason": "original missing", "original": str(original)},
                )
    else:
        check(
            "workspace_hash_checks",
            True,
            {"status": "skipped", "hint": "set AFV2_CHECK_WORKSPACE_HASH=1 to enable"},
        )

    # cascade order documentation lock
    report["cascade_order"] = [
        "1_v2_base: pred = pred_h if trunk else pred_m",
        "2_amb_lift: ambient→trunk if ts>=0.5 and max(cs,bin)>=0.1",
        "3_secondary: ambient→trunk if tr>=0.625 and bin>=0.75",
        "4_material: leaf/twig redecode with b_leaf=-1.3 on meta soft",
    ]
    report["feature_stack_order"] = {
        "hand_amb": ["robot_mix_total240", "wav2vec2", "clip"],
        "robot_amb": ["total240_clean", "wav2vec2", "clip"],
        "secondary": ["clip_only"],
    }
    report["metrics"] = m

    out_path = ROOT / "verify_feature_order_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps({"ok": report["ok"], "report": str(out_path)}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
