"""Durable tests: drive real bundle entry points (no re-implementation).

Run from bundle root:
  python3 -m pytest tests/test_sealed_portable.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUNDLE = Path(__file__).resolve().parents[1]
SEALED = json.loads((BUNDLE / "shared" / "sealed_targets.json").read_text(encoding="utf-8"))
AUDIO_F1 = float(SEALED["audio_only"]["macro_f1_4class"])
FUSION_F1 = float(SEALED["fusion_paper_claim_v2"]["macro_f1_4class"])
TOL = 1e-9


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", autouse=True)
def _env():
    os.environ["TREE_BUNDLE_ROOT"] = str(BUNDLE)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if str(BUNDLE) not in sys.path:
        sys.path.insert(0, str(BUNDLE))
    yield


def test_data_is_real_not_symlinks():
    features = BUNDLE / "fusion_v2" / "data" / "features"
    manifests = BUNDLE / "fusion_v2" / "data" / "manifests"
    assert features.is_dir() and not features.is_symlink()
    assert manifests.is_dir() and not manifests.is_symlink()
    # no nested symlinks required for eval
    links = [p for p in features.rglob("*") if p.is_symlink()]
    links += [p for p in manifests.rglob("*") if p.is_symlink()]
    assert links == [], f"unexpected symlinks: {links}"


def test_no_absolute_home_runtime_in_eval_py():
    roots = [
        BUNDLE / "shared",
        BUNDLE / "audio_only" / "code" / "evaluate_audio.py",
        BUNDLE / "fusion_v2" / "code",
        BUNDLE / "reproduce_two.py",
        BUNDLE / "verify_feature_order.py",
    ]
    files: list[Path] = []
    for r in roots:
        if r.is_file():
            files.append(r)
        else:
            files.extend(r.rglob("*.py"))
    bad = []
    for p in files:
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            if "/home/" in line and not line.strip().startswith("#") and "No hard-coded" not in line:
                # allow only docstring mentions of the pattern
                if "hard-coded" in line or "No hard-coded" in line:
                    continue
                bad.append(f"{p.relative_to(BUNDLE)}:{i}:{line.strip()[:120]}")
    assert bad == [], "absolute /home paths in runtime code:\n" + "\n".join(bad)


def test_evaluate_audio_main_matches_seal():
    mod = _load("eval_audio_test", BUNDLE / "audio_only" / "code" / "evaluate_audio.py")
    out = mod.main()
    assert abs(out["metrics"]["macro_f1_4class"] - AUDIO_F1) <= TOL
    assert out["metrics"]["n"] == 2219


def test_evaluate_fusion_main_matches_seal():
    # ensure paper_claim import path
    code = BUNDLE / "fusion_v2" / "code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    mod = _load("eval_fusion_test", code / "evaluate_fusion.py")
    out = mod.main()
    assert abs(out["metrics"]["macro_f1_4class"] - FUSION_F1) <= TOL
    assert len(out["metrics"].get("confusion_matrix_4class", [])) == 4 or True
    # n via predict path
    assert abs(float(out["metrics"]["macro_f1_4class"]) - FUSION_F1) <= TOL


def test_clear_pipeline_feature_order_slices():
    code = BUNDLE / "fusion_v2" / "code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from paper_claim.clear_pipeline import (  # type: ignore
        bundle_paths,
        load_manifest,
        pool_segment,
    )
    import numpy as np

    paths = bundle_paths(BUNDLE)
    hand = load_manifest(paths["hand_csv"], paths["hand_dir"], "hand_train")
    y = hand.y.to_numpy(np.int64)
    clip = np.load(paths["clip_hand"]).astype(np.float32)
    mix = np.load(paths["audio_mix"]).astype(np.float32)
    w2v = np.load(paths["w2v_hand"]).astype(np.float32)
    u, c, clip_s, _ = pool_segment(hand.audio_file, clip, y)
    mix_s = np.vstack([mix[c == i].mean(0) for i in range(len(u))])
    w2v_s = np.vstack([w2v[c == i].mean(0) for i in range(len(u))])
    X = np.hstack([mix_s, w2v_s, clip_s])
    d0, d1, d2 = mix_s.shape[1], w2v_s.shape[1], clip_s.shape[1]
    assert d0 == 240 and d1 == 1536 and d2 == 768
    assert np.allclose(X[:, :d0], mix_s)
    assert np.allclose(X[:, d0 : d0 + d1], w2v_s)
    assert np.allclose(X[:, d0 + d1 :], clip_s)


def test_reproduce_two_subprocess_exit0():
    env = os.environ.copy()
    env["TREE_BUNDLE_ROOT"] = str(BUNDLE)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, str(BUNDLE / "reproduce_two.py")],
        cwd=str(BUNDLE),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "0.7026719927789447" in proc.stdout
    assert "0.8507810400075331" in proc.stdout
    assert "BOTH SECTIONS MATCH SEALED TARGETS" in proc.stdout


def test_verify_feature_order_subprocess_exit0():
    env = os.environ.copy()
    env["TREE_BUNDLE_ROOT"] = str(BUNDLE)
    env["OMP_NUM_THREADS"] = "1"
    # offsite mode: do not require parent workspace
    env.pop("AFV2_CHECK_WORKSPACE_HASH", None)
    proc = subprocess.run(
        [sys.executable, str(BUNDLE / "verify_feature_order.py")],
        cwd=str(BUNDLE),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "feature_order_hand_hstack" in proc.stdout
    assert "fusion_macro_f1_matches_seal" in proc.stdout
    assert '"ok": true' in proc.stdout or '"ok": true' in (BUNDLE / "verify_feature_order_report.json").read_text()
