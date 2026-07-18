"""Pure single-shot paper claim protocol guards.

Architecture:
  1) RECIPE.json is frozen/pre-registered (hashed into the lock).
  2) Hand selection only → selection_lock.json (test_loaded=false).
  3) Final test opens robot once → writes CLAIM_SEAL.json (irreversible for claim dir).
  4) Any second open of robot for the same claim_out is refused.

Exploratory multi-peek under outputs/audio_feature_benchmarks/multimodal_085_* is
explicitly non-claim. Claim dirs: outputs/paper_claim_v1/, outputs/paper_claim_v2/, …
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import multimodal_085_protocol as proto

REPO = Path(__file__).resolve().parent
RECIPE_PATH = REPO / "paper_claim" / "RECIPE.json"
RECIPE_V2_PATH = REPO / "paper_claim" / "RECIPE_v2_simple.json"
DEFAULT_CLAIM_OUT = REPO / "outputs" / "paper_claim_v1"


def load_recipe(path: Path | None = None) -> dict:
    path = Path(path or os.environ.get("PAPER_CLAIM_RECIPE") or RECIPE_PATH)
    recipe = json.loads(path.read_text())
    return recipe


def recipe_sha256(recipe: dict | None = None, path: Path | None = None) -> str:
    if recipe is None:
        raw = Path(path or os.environ.get("PAPER_CLAIM_RECIPE") or RECIPE_PATH).read_bytes()
    else:
        raw = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def claim_out_dir(recipe: dict | None = None) -> Path:
    recipe = recipe or load_recipe()
    env = os.environ.get("PAPER_CLAIM_OUT")
    if env:
        return Path(env)
    rel = recipe.get("artifacts", {}).get("claim_out", "outputs/paper_claim_v1")
    return REPO / rel


def seal_path(claim_out: Path | None = None) -> Path:
    return Path(claim_out or claim_out_dir()) / "CLAIM_SEAL.json"


def selection_lock_path(claim_out: Path | None = None) -> Path:
    return Path(claim_out or claim_out_dir()) / "selection_lock.json"


def assert_not_sealed(claim_out: Path | None = None) -> None:
    p = seal_path(claim_out)
    if p.exists() and os.environ.get("PAPER_CLAIM_FORCE_REOPEN") != "1":
        raise RuntimeError(
            f"Paper claim already sealed at {p}. Pure single-shot forbids a second "
            f"robot open. Set PAPER_CLAIM_FORCE_REOPEN=1 only for non-claim diagnostics."
        )


def assert_selection_lock_clean(lock: dict) -> None:
    if lock.get("test_loaded"):
        raise AssertionError("selection lock already test-loaded")
    inv = lock.get("invariants", {})
    if not inv.get("robot_not_used_in_selection", True):
        raise AssertionError("lock invariants claim robot used in selection")
    if inv.get("filename_class_features"):
        raise AssertionError("filename class features forbidden")
    if inv.get("robot_uda"):
        raise AssertionError("robot UDA forbidden")


def write_hand_selection_lock(
    claim_out: Path,
    payload: dict,
    recipe: dict,
    recipe_path: Path | None = None,
) -> dict:
    """Write selection lock only if claim not sealed and robot never loaded."""
    claim_out = Path(claim_out)
    claim_out.mkdir(parents=True, exist_ok=True)
    assert_not_sealed(claim_out)
    # refuse overwrite if a sealed final already exists
    if (claim_out / "final_test_metrics.json").exists() and os.environ.get(
        "PAPER_CLAIM_FORCE_REOPEN"
    ) != "1":
        raise RuntimeError(
            f"final_test_metrics.json already exists under {claim_out}; "
            "single-shot claim path refuses re-selection after test open"
        )
    rpath = Path(
        recipe_path
        or os.environ.get("PAPER_CLAIM_RECIPE")
        or RECIPE_PATH
    )
    lock = dict(payload)
    lock["recipe_id"] = recipe["recipe_id"]
    lock["recipe_sha256"] = recipe_sha256(path=rpath)
    try:
        lock["recipe_path"] = str(rpath.resolve().relative_to(REPO))
    except ValueError:
        lock["recipe_path"] = str(rpath)
    lock["claim_type"] = "pure_single_shot"
    lock["test_loaded"] = False
    lock["selection_data"] = "hand/default only"
    lock["group_column"] = recipe["data"]["group_column"]
    lock["invariants"] = {
        "robot_not_used_in_selection": True,
        "filename_class_features": False,
        "no_test_hp_tuning": True,
        "robot_uda": False,
        "multi_open_robot_for_claim": False,
        "pure_single_shot_architecture": True,
        "exploratory_multimodal_085_peeks_excluded_from_claim": True,
    }
    lock["created_utc"] = datetime.now(timezone.utc).isoformat()
    out = claim_out / "selection_lock.json"
    out.write_text(json.dumps(lock, indent=2, default=float))
    # also copy recipe for provenance
    (claim_out / "RECIPE.frozen.json").write_text(rpath.read_text())
    return lock


def load_hand_selection_lock(
    claim_out: Path | None = None,
    recipe_path: Path | None = None,
) -> dict:
    claim_out = Path(claim_out or claim_out_dir())
    assert_not_sealed(claim_out)
    path = selection_lock_path(claim_out)
    if not path.exists():
        raise FileNotFoundError(f"missing selection lock: {path}")
    lock = json.loads(path.read_text())
    assert_selection_lock_clean(lock)
    # recipe integrity — prefer lock-recorded path, then env/arg
    rpath = Path(
        recipe_path
        or os.environ.get("PAPER_CLAIM_RECIPE")
        or (REPO / lock["recipe_path"] if lock.get("recipe_path") else RECIPE_PATH)
    )
    expected = recipe_sha256(path=rpath)
    if lock.get("recipe_sha256") != expected:
        raise AssertionError(
            f"recipe hash mismatch: lock={lock.get('recipe_sha256')} now={expected}. "
            "Do not edit RECIPE after selection without a new claim version."
        )
    return lock


def seal_claim(
    claim_out: Path,
    lock: dict,
    metrics: dict,
    extra: dict | None = None,
) -> dict:
    """Irreversible seal after the single robot open."""
    claim_out = Path(claim_out)
    claim_out.mkdir(parents=True, exist_ok=True)
    if seal_path(claim_out).exists() and os.environ.get("PAPER_CLAIM_FORCE_REOPEN") != "1":
        raise RuntimeError("claim already sealed")
    seal = {
        "sealed": True,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "recipe_id": lock.get("recipe_id"),
        "recipe_sha256": lock.get("recipe_sha256"),
        "selected_candidate": lock.get("selected_candidate"),
        "final_test_macro_f1": metrics.get("metrics", {}).get("macro_f1_4class"),
        "protocol": "paper_claim_pure_single_shot_v1",
        "invariants": {
            "robot_opened_once": True,
            "no_reopen_without_force_env": True,
            "selection_was_hand_only": True,
            "filename_class_features": False,
            "robot_uda": False,
            "no_test_hp_tuning": True,
        },
        "note": (
            "This seal marks the only allowed robot open for this claim directory. "
            "Further peeks are non-claim diagnostics."
        ),
    }
    if extra:
        seal.update(extra)
    seal_path(claim_out).write_text(json.dumps(seal, indent=2, default=float))
    # mark lock as test-loaded (copy, do not allow re-use via load_hand_selection_lock)
    lock_after = dict(lock)
    lock_after["test_loaded"] = True
    lock_after["final_test_macro_f1"] = seal["final_test_macro_f1"]
    lock_after["sealed"] = True
    (claim_out / "selection_lock_after_test.json").write_text(
        json.dumps(lock_after, indent=2, default=float)
    )
    return seal


def parse_linspace(spec: str) -> np.ndarray:
    """Parse 'linspace(a, b, n)' strings from recipe grid."""
    spec = spec.strip()
    assert spec.startswith("linspace(") and spec.endswith(")")
    a, b, n = spec[len("linspace(") : -1].split(",")
    return np.linspace(float(a), float(b), int(n))


def normalize(p: np.ndarray) -> np.ndarray:
    return proto.normalize(p)


def fast_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return proto.fast_macro_f1(y, pred)


def metrics_bundle(y: np.ndarray, pred: np.ndarray) -> dict:
    return proto.metrics_bundle(y, pred)
