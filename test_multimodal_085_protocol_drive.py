"""Protocol tests for multimodal 0.85 drive: lock helpers + real metrics artifact."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import multimodal_085_protocol as proto

ROOT = Path(__file__).resolve().parent
BEST_METRICS = (
    ROOT
    / "outputs/audio_feature_benchmarks/multimodal_085_best_effort_group_selection"
    / "BEST_STRICT_final_test_metrics.json"
)
BEST_LOCK = (
    ROOT
    / "outputs/audio_feature_benchmarks/multimodal_085_best_effort_group_selection"
    / "selection_lock_best_0827.json"
)
V2_METRICS = (
    ROOT
    / "outputs/audio_feature_benchmarks/segment_rule_stack_v2_group_selection"
    / "segment_rule_stack_v2_final_test_metrics.json"
)


class TestMultimodal085ProtocolDrive(unittest.TestCase):
    def test_write_selection_lock_forces_test_not_loaded(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "selection_lock.json"
            lock = proto.write_selection_lock(
                path,
                {
                    "protocol": "unit_test",
                    "selected_candidate": {"alpha": 0.1},
                    "test_loaded": True,  # must be overwritten
                },
            )
            self.assertFalse(lock["test_loaded"])
            disk = json.loads(path.read_text())
            self.assertFalse(disk["test_loaded"])
            self.assertEqual(disk["selection_data"], "hand/default only")
            self.assertEqual(disk["group_column"], "specimen_group")
            self.assertTrue(disk["invariants"]["robot_not_used_in_selection"])
            self.assertFalse(disk["invariants"]["filename_class_features"])
            self.assertTrue(disk["invariants"]["no_test_hp_tuning"])

    def test_load_selection_lock_rejects_test_loaded(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "selection_lock.json"
            path.write_text(json.dumps({"test_loaded": True, "selected_candidate": {}}))
            with self.assertRaises(AssertionError):
                proto.load_selection_lock(path)

    def test_metrics_bundle_macro_f1_path(self):
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        pred = np.array([0, 1, 2, 3, 0, 1, 2, 2], dtype=np.int64)
        bundle = proto.metrics_bundle(y, pred)
        self.assertIn("macro_f1_4class", bundle["metrics"])
        self.assertEqual(bundle["n"], 8)
        self.assertEqual(len(bundle["confusion_matrix_4class"]), 4)
        # perfect-ish except one twig->trunk
        self.assertGreater(bundle["metrics"]["macro_f1_4class"], 0.8)
        self.assertLess(bundle["metrics"]["macro_f1_4class"], 1.0)

    def test_best_strict_artifact_exists_and_has_invariants(self):
        self.assertTrue(BEST_METRICS.is_file(), f"missing {BEST_METRICS}")
        self.assertTrue(BEST_LOCK.is_file(), f"missing {BEST_LOCK}")
        metrics = json.loads(BEST_METRICS.read_text())
        lock = json.loads(BEST_LOCK.read_text())
        self.assertFalse(lock.get("test_loaded"))
        self.assertIn("hand", str(lock.get("selection_data", "")).lower())
        f1 = float(metrics["metrics"]["macro_f1_4class"])
        self.assertGreater(f1, 0.80)
        # session best is ~0.827; assert lift over v2 baseline
        v2 = json.loads(V2_METRICS.read_text())
        v2_f1 = float(v2["metrics"]["macro_f1_4class"])
        self.assertGreater(f1, v2_f1)
        inv = metrics["invariants"]
        self.assertTrue(inv.get("selection_used_hand_only"))
        self.assertTrue(inv.get("no_test_hp_tuning"))
        self.assertFalse(inv.get("filename_class_features"))
        self.assertFalse(inv.get("robot_uda"))
        self.assertIn("trunk", metrics["per_class_4class"])
        self.assertEqual(len(metrics["confusion_matrix_4class"]), 4)

    def test_fast_macro_f1_matches_sklearn_on_bundle(self):
        from sklearn.metrics import f1_score

        y = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        pred = np.array([0, 1, 1, 1, 2, 3, 3, 3], dtype=np.int64)
        self.assertAlmostEqual(
            proto.fast_macro_f1(y, pred),
            float(f1_score(y, pred, average="macro", zero_division=0)),
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
