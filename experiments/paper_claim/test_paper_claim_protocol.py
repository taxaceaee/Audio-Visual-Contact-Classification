"""Tests for pure single-shot paper claim guards."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import paper_claim_protocol as claim


def test_recipe_loads_and_hashes():
    recipe = claim.load_recipe()
    assert recipe["recipe_id"]
    h1 = claim.recipe_sha256()
    h2 = claim.recipe_sha256()
    assert h1 == h2
    assert len(h1) == 64


def test_selection_lock_refuses_test_loaded(tmp_path, monkeypatch):
    recipe = claim.load_recipe()
    # point claim out to tmp
    monkeypatch.setenv("PAPER_CLAIM_OUT", str(tmp_path))
    lock = claim.write_hand_selection_lock(
        tmp_path,
        {"protocol": "test", "selected_candidate": {"mode": "none", "T": 1.0, "b_leaf": 0, "b_trunk": 0, "b_twig": 0}},
        recipe,
    )
    assert lock["test_loaded"] is False
    loaded = claim.load_hand_selection_lock(tmp_path)
    assert loaded["recipe_sha256"] == claim.recipe_sha256()
    # mark test loaded
    loaded["test_loaded"] = True
    (tmp_path / "selection_lock.json").write_text(json.dumps(loaded))
    with pytest.raises(AssertionError):
        claim.load_hand_selection_lock(tmp_path)


def test_seal_blocks_second_open(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_CLAIM_OUT", str(tmp_path))
    monkeypatch.delenv("PAPER_CLAIM_FORCE_REOPEN", raising=False)
    recipe = claim.load_recipe()
    lock = claim.write_hand_selection_lock(
        tmp_path,
        {"protocol": "test", "selected_candidate": {"mode": "none"}},
        recipe,
    )
    claim.seal_claim(tmp_path, lock, {"metrics": {"macro_f1_4class": 0.5}})
    with pytest.raises(RuntimeError, match="already sealed"):
        claim.assert_not_sealed(tmp_path)
    with pytest.raises(RuntimeError, match="already sealed"):
        claim.load_hand_selection_lock(tmp_path)


def test_parse_linspace():
    arr = claim.parse_linspace("linspace(-1.6, 0.2, 10)")
    assert len(arr) == 10
    assert abs(arr[0] + 1.6) < 1e-9
    assert abs(arr[-1] - 0.2) < 1e-9


def test_recipe_forbids_listed():
    recipe = claim.load_recipe()
    forb = recipe["forbidden"]
    assert forb["robot_in_selection"] is True
    assert forb["test_hp_tuning"] is True
    assert forb["multi_open_robot_for_claim"] is True
