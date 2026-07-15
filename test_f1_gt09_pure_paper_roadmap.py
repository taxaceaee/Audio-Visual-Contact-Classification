"""Structural + residual-budget checks for the pure-paper F1>0.9 roadmap.

Drives real sealed claim metrics on disk and the shipped roadmap document.
Does not open robot or retune HPs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
ROADMAP = ROOT / "docs" / "MULTIMODAL_F1_GT09_PURE_PAPER_ROADMAP.md"
SEALED_V2 = ROOT / "outputs" / "paper_claim_v2" / "final_test_metrics.json"
SEALED_V1 = ROOT / "outputs" / "paper_claim_v1" / "final_test_metrics.json"
CLAIM_SEAL = ROOT / "outputs" / "paper_claim_v2" / "CLAIM_SEAL.json"

# Sealed robot CM (rows=true) from paper_claim_v2 / evaluation_report
SEALED_CM = np.array(
    [
        [1131, 0, 1, 0],
        [2, 220, 20, 51],
        [45, 0, 340, 76],
        [28, 0, 18, 287],
    ],
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


def _fix(cm: np.ndarray, pairs) -> np.ndarray:
    c = cm.copy()
    for t, p in pairs:
        n = int(c[t, p])
        c[t, p] = 0
        c[t, t] += n
    return c


def test_sealed_v2_baseline_macro_f1():
    assert SEALED_V2.is_file(), "missing sealed paper_claim_v2 metrics"
    d = json.loads(SEALED_V2.read_text())
    f1 = float(d["metrics"]["macro_f1_4class"])
    assert 0.850 < f1 < 0.852, f"unexpected sealed v2 F1: {f1}"
    assert abs(f1 - _cm_f1(SEALED_CM)) < 1e-9
    assert d["n"] == 2219
    assert d["split"] == "robot_test_final"


def test_sealed_v1_best_absolute_still_below_0_9():
    assert SEALED_V1.is_file()
    f1 = float(json.loads(SEALED_V1.read_text())["metrics"]["macro_f1_4class"])
    assert 0.853 < f1 < 0.854
    assert f1 < 0.9


def test_claim_seal_pure_paper_invariants():
    seal = json.loads(CLAIM_SEAL.read_text())
    inv = seal["invariants"]
    assert inv["selection_was_hand_only"] is True
    assert inv["filename_class_features"] is False
    assert inv["robot_uda"] is False
    assert inv["no_test_hp_tuning"] is True
    assert inv["robot_opened_once"] is True


def test_contact_only_oracle_cannot_reach_0_9():
    """On sealed CM, perfect contact residual recovery stays below 0.9."""
    contact_pairs = [(2, 0), (3, 0), (1, 0), (0, 2)]
    f1 = _cm_f1(_fix(SEALED_CM, contact_pairs))
    assert f1 < 0.9
    assert f1 > 0.88  # still a real lift


def test_material_residual_oracle_exceeds_0_9():
    """Wood/leaf material residual alone has information room above 0.9."""
    material_pairs = [(2, 3), (1, 3), (3, 2), (1, 2)]
    f1 = _cm_f1(_fix(SEALED_CM, material_pairs))
    assert f1 > 0.9
    assert f1 > 0.95


def test_cheapest_top_modes_combo_can_cross_0_9():
    """~15 trunk→twig + all leaf→twig is enough in oracle combo space."""
    c = SEALED_CM.copy()
    # fix 15 trunk→twig
    c[2, 3] -= 15
    c[2, 2] += 15
    # fix all 51 leaf→twig
    n = int(c[1, 3])
    c[1, 3] = 0
    c[1, 1] += n
    assert _cm_f1(c) > 0.9


def test_roadmap_document_exists_and_covers_acceptance():
    assert ROADMAP.is_file(), f"missing roadmap: {ROADMAP}"
    text = ROADMAP.read_text()
    # target + baseline
    assert re.search(r"0\.9|0\.900", text)
    assert "0.850781" in text or "0.8508" in text
    assert "0.853226" in text or "0.8532" in text
    # pure-paper hard requirements
    for phrase in [
        "hand",
        "one-shot",
        "robot UDA",
        "filename",
        "test HP",
        "specimen_group",
    ]:
        assert phrase.lower() in text.lower(), f"missing pure-paper phrase: {phrase}"
    # residual modes
    for mode in ["trunk→twig", "trunk→ambient", "leaf→twig", "twig→ambient"]:
        assert mode in text or mode.replace("→", "->") in text
    # distinguish frozen reweight vs representation change
    assert "0.879" in text or "frozen" in text.lower()
    assert "representation" in text.lower() or "P0.1" in text
    # no false claim of new seal in this analysis
    assert "No new sealed robot run was executed" in text
    # ranked paths present
    assert "P0.1" in text and "P0.2" in text and "P0.3" in text
    assert "Kill criteria" in text or "kill criteria" in text.lower()
