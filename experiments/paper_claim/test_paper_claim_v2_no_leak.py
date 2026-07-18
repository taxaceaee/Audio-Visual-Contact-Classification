"""Leakage / purity checks for paper claim v2 simple path."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import paper_claim_protocol as claim

REPO = Path(__file__).resolve().parent
SEL = REPO / "run_paper_claim_v2_hand_selection.py"
FT = REPO / "run_paper_claim_v2_final_test_once.py"
RECIPE = claim.RECIPE_V2_PATH


def test_recipe_v2_loads():
    r = claim.load_recipe(RECIPE)
    assert r["recipe_id"] == "paper_claim_v2_simple_leaf_bias"
    assert r["material_decode"]["free_params"] == ["b_leaf"]
    assert r["frozen_contact_stack"]["primary"]["do_tt"] is False
    assert r["artifacts"]["claim_out"] == "outputs/paper_claim_v2"
    assert r["forbidden"]["robot_in_selection"] is True
    assert r["forbidden"]["test_hp_tuning"] is True


def test_selection_script_never_loads_robot():
    src = SEL.read_text()
    # AST-level: no string literals that load robot test
    for bad in (
        "robot_test",
        "audio_visual_dataset_robo",
        "robot_test_X",
        "robot_test/",
    ):
        assert bad not in src, f"selection script mentions {bad}"
    assert "hand_train" in src
    assert "StratifiedGroupKFold" in src


def test_final_loads_robot_only_after_lock_api():
    src = FT.read_text()
    assert "load_hand_selection_lock" in src
    assert "assert_not_sealed" in src
    assert "seal_claim" in src
    assert "robot_test" in src  # allowed in final only


def test_selection_parses_as_valid_python():
    ast.parse(SEL.read_text())
    ast.parse(FT.read_text())


def test_no_filename_class_feature_extraction():
    for path in (SEL, FT):
        src = path.read_text()
        # forbid patterns that inject class from path into model features
        assert not re.search(
            r"parse_class|label_from_name|class_from_stem|y_from_filename",
            src,
            re.I,
        )
        # group keys must come from sel.seg/spec strip only
        if path == SEL:
            assert "sel.spec" in src and "sel.seg" in src
