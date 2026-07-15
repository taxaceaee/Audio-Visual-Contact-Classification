"""Drive shipped gt09 pure-paper claim entry points and honesty checks.

Does not claim F1>0.9. Asserts protocol, sealed attempt metrics, residual budget,
and second-open refusal on the real claim directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
CLAIM = ROOT / "outputs" / "paper_claim_gt09"
RECIPE = ROOT / "paper_claim" / "RECIPE_gt09_material_multibb.json"
SEALED_V2 = ROOT / "outputs" / "paper_claim_v2" / "final_test_metrics.json"

SEALED_CM = np.array(
    [[1131, 0, 1, 0], [2, 220, 20, 51], [45, 0, 340, 76], [28, 0, 18, 287]],
    dtype=int,
)


def _cm_f1(cm: np.ndarray) -> float:
    yt, yp = [], []
    for t in range(4):
        for p in range(4):
            n = int(cm[t, p])
            yt.extend([t] * n)
            yp.extend([p] * n)
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def _fix(cm, pairs):
    c = cm.copy()
    for t, p in pairs:
        n = int(c[t, p])
        c[t, p] = 0
        c[t, t] += n
    return c


def test_claim_artifacts_exist():
    for name in [
        "selection_lock.json",
        "selection_lock_after_test.json",
        "final_test_metrics.json",
        "CLAIM_SEAL.json",
        "hand_selection_leaderboard.csv",
        "RECIPE.frozen.json",
    ]:
        assert (CLAIM / name).is_file(), name


def test_selection_lock_was_hand_only():
    lock = json.loads((CLAIM / "selection_lock.json").read_text())
    # After test, primary lock file may still show pre-test payload; after_test has test_loaded
    after = json.loads((CLAIM / "selection_lock_after_test.json").read_text())
    assert after.get("test_loaded") is True
    assert after.get("selection_data") == "hand/default only" or lock.get("selection_data") == "hand/default only"
    inv = lock.get("invariants", {})
    assert inv.get("filename_class_features") is False
    assert inv.get("robot_uda") is False
    assert inv.get("no_test_hp_tuning") is True
    assert inv.get("robot_not_used_in_selection") is True
    assert "selected_candidate" in lock
    assert lock.get("recipe_sha256")


def test_seal_invariants_and_one_shot():
    seal = json.loads((CLAIM / "CLAIM_SEAL.json").read_text())
    assert seal["sealed"] is True
    inv = seal["invariants"]
    assert inv["robot_opened_once"] is True
    assert inv["selection_was_hand_only"] is True
    assert inv["filename_class_features"] is False
    assert inv["robot_uda"] is False
    assert inv["no_test_hp_tuning"] is True
    assert inv["no_reopen_without_force_env"] is True


def test_final_metrics_real_robot_split_not_gt_0_9():
    """Honest: this pure attempt did not clear 0.9; metrics come from sealed file."""
    m = json.loads((CLAIM / "final_test_metrics.json").read_text())
    assert m["split"] == "robot_test_final"
    assert m["n"] == 2219
    f1 = float(m["metrics"]["macro_f1_4class"])
    assert f1 == pytest.approx(float(json.loads((CLAIM / "CLAIM_SEAL.json").read_text())["final_test_macro_f1"]))
    assert f1 < 0.9  # failed pure attempt — do not invent success
    assert m["invariants"]["selection_used_hand_only"] is True
    assert m["invariants"]["robot_uda"] is False
    assert m["invariants"]["filename_class_features"] is False


def test_sealed_v2_still_below_0_9():
    f1 = float(json.loads(SEALED_V2.read_text())["metrics"]["macro_f1_4class"])
    assert 0.85 < f1 < 0.86
    assert f1 < 0.9


def test_contact_only_oracle_below_0_9():
    f1 = _cm_f1(_fix(SEALED_CM, [(2, 0), (3, 0), (1, 0), (0, 2)]))
    assert f1 < 0.9


def test_material_oracle_above_0_9_room():
    f1 = _cm_f1(_fix(SEALED_CM, [(2, 3), (1, 3), (3, 2), (1, 2)]))
    assert f1 > 0.9


def test_second_open_refused_without_force_env():
    env = os.environ.copy()
    env["PAPER_CLAIM_RECIPE"] = str(RECIPE)
    env["PAPER_CLAIM_OUT"] = str(CLAIM)
    env.pop("PAPER_CLAIM_FORCE_REOPEN", None)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run_paper_claim_gt09_final_test_once.py")],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    blob = (proc.stdout or "") + (proc.stderr or "")
    assert "sealed" in blob.lower() or "single-shot" in blob.lower() or "refuses" in blob.lower()


def test_gt09_core_segment_key_aligns_v2():
    from paper_claim import gt09_core as core

    z = np.load(core.V2_BASE, allow_pickle=True)
    ids = set(z["segment_ids"].astype(str))
    paths, _ = core.load_robot_window()
    keys = {core.segment_key(p) for p in paths.astype(str)}
    assert keys == ids


def test_hand_selection_script_is_shipped():
    assert (ROOT / "run_paper_claim_gt09_hand_selection.py").is_file()
    assert (ROOT / "run_paper_claim_gt09_final_test_once.py").is_file()
    assert RECIPE.is_file()
