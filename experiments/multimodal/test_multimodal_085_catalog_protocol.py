"""Verify MULTIMODAL_085 strict catalog artifact + baseline metrics (no robot tuning).

Drives real on-disk artifacts and catalog content required by the analysis goal.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "MULTIMODAL_085_STRICT_CATALOG.md"
ATTEMPT_LOG = ROOT / "MULTIMODAL_085_ATTEMPT_LOG.md"
V2_METRICS = (
    ROOT
    / "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection"
    / "segment_rule_stack_v2_final_test_metrics.json"
)
V2_LOCK = (
    ROOT
    / "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection"
    / "selection_lock.json"
)


class TestMultimodal085CatalogProtocol(unittest.TestCase):
    def test_catalog_exists_and_states_target_and_constraints(self):
        self.assertTrue(CATALOG.is_file(), f"missing catalog: {CATALOG}")
        text = CATALOG.read_text(encoding="utf-8")
        self.assertIn("0.85", text)
        self.assertRegex(text, re.compile(r"no data leakage|No data leakage", re.I))
        self.assertRegex(
            text,
            re.compile(
                r"no HP tuning on robot/test|No HP tuning on robot/test|no hyperparameter tuning on robot",
                re.I,
            ),
        )
        # multiple method families
        for section in (
            "### A. Protocol hygiene",
            "### B. Audio domain-robust",
            "### C. Image domain-robust",
            "### D. Hierarchical fusion",
            "### E. True joint multimodal",
            "### F. Controlled ensembling",
            "### G. Data / problem structure",
        ):
            self.assertIn(section, text, f"missing family section: {section}")
        self.assertIn("Forbidden / non-strict", text)
        self.assertIn("hand/default only", text)
        self.assertIn("selection_lock", text)

    def test_forbidden_section_lists_test_tuning_and_filename_leak(self):
        text = CATALOG.read_text(encoding="utf-8")
        # locate forbidden section
        idx = text.find("## 5. Forbidden")
        self.assertGreaterEqual(idx, 0)
        forbidden = text[idx : idx + 2500]
        self.assertRegex(forbidden, re.compile(r"robot/test|robot", re.I))
        self.assertRegex(forbidden, re.compile(r"filename", re.I))
        self.assertRegex(forbidden, re.compile(r"UDA|unlabeled|adaptation", re.I))
        # must not recommend filename priors under strict catalog body as allowed
        self.assertIn("must not be sold as the strict", forbidden.lower())

    def test_sample_methods_state_hand_only_and_anti_leak(self):
        text = CATALOG.read_text(encoding="utf-8")
        # sample three method rows / entries
        samples = ["B1", "D3", "E1"]
        for mid in samples:
            self.assertIn(mid, text)
        # table headers / columns present
        self.assertIn("Hand-only selection", text)
        self.assertIn("Anti-leak", text)
        # explicit phrases for the three families
        self.assertIn("Worst-view", text)
        self.assertIn("Nested", text)
        self.assertIn("specimen", text.lower())

    def test_v2_baseline_metrics_match_catalog_numbers(self):
        self.assertTrue(V2_METRICS.is_file(), f"missing baseline metrics: {V2_METRICS}")
        data = json.loads(V2_METRICS.read_text(encoding="utf-8"))
        m = data["metrics"]
        f1 = float(m["macro_f1_4class"])
        self.assertAlmostEqual(f1, 0.7966563229211119, places=6)
        trunk_rec = float(data["per_class_4class"]["trunk"]["recall"])
        self.assertAlmostEqual(trunk_rec, 0.5683297180043384, places=6)
        cm = data["confusion_matrix_4class"]
        # trunk→ambient count
        self.assertEqual(cm[2][0], 123)
        # invariants claim hand-only selection
        inv = data.get("invariants", {})
        self.assertTrue(inv.get("selection_used_hand_only"))
        self.assertTrue(inv.get("test_loaded_after_lock"))

        catalog = CATALOG.read_text(encoding="utf-8")
        self.assertIn("0.796656", catalog)
        self.assertIn("123", catalog)
        self.assertIn("0.5683", catalog)

    def test_v2_selection_lock_was_hand_only_pattern(self):
        self.assertTrue(V2_LOCK.is_file())
        lock = json.loads(V2_LOCK.read_text(encoding="utf-8"))
        # historical lock may have been updated; require selection_data hand
        self.assertIn("hand", str(lock.get("selection_data", "")).lower())
        self.assertIn("specimen_group", str(lock.get("group_column", "specimen_group")))

    def test_attempt_log_cross_links_catalog_goal(self):
        self.assertTrue(ATTEMPT_LOG.is_file())
        log = ATTEMPT_LOG.read_text(encoding="utf-8")
        self.assertIn("0.85", log)
        self.assertIn("0.796656", log)
        self.assertIn("hand/default only", log)


if __name__ == "__main__":
    unittest.main()
