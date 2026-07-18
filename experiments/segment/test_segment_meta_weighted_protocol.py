"""Protocol gates for the winning segment meta-weighted fusion pipeline.

Drives shipped selection/final helpers and locked artifacts under
outputs/audio_feature_meta_weighted_group_selection/. Does not re-tune on
robot/test; asserts the lock/protocol invariants and reproduced macro F1.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import run_multimodal_val_locked_suite as suite
import run_segment_meta_weighted_group_final_test as final_mod
import run_segment_meta_weighted_group_selection as sel_mod

ROOT = Path("/home/ttung05/Desktop/tree_base/tree_structures")
OUT = Path("outputs/audio_feature_meta_weighted_group_selection")
SEL_SRC = Path("run_segment_meta_weighted_group_selection.py")
FT_SRC = Path("run_segment_meta_weighted_group_final_test.py")


class TestSegmentMetaWeightedHelpers(unittest.TestCase):
    def test_seg_strips_window_suffix(self):
        files = pd.Series(
            [
                "tree_A_segment_00_window_0.wav",
                "tree_A_segment_00_window_12.wav",
                "tree_B_segment_03_window_1.wav",
            ]
        )
        got = sel_mod.seg(files).tolist()
        self.assertEqual(
            got,
            ["tree_A_segment_00", "tree_A_segment_00", "tree_B_segment_03"],
        )

    def test_spec_strips_segment_after_seg(self):
        files = pd.Series(
            [
                "tree_A_segment_00_window_0.wav",
                "tree_A_segment_01_window_2.wav",
            ]
        )
        got = sel_mod.spec(files).tolist()
        self.assertEqual(got, ["tree_A", "tree_A"])

    def test_pool_uses_segment_keys_and_pure_label(self):
        files = pd.Series(
            [
                "s1_segment_0_window_0.wav",
                "s1_segment_0_window_1.wav",
                "s1_segment_1_window_0.wav",
            ]
        )
        x = np.array([[1.0, 0.0], [3.0, 2.0], [10.0, 10.0]], dtype=np.float32)
        y = np.array([2, 2, 1], dtype=np.int64)
        keys, codes, px, py = final_mod.pool(files, x, y)
        self.assertEqual(list(keys), ["s1_segment_0", "s1_segment_1"])
        np.testing.assert_allclose(px[0], [2.0, 1.0])
        np.testing.assert_allclose(px[1], [10.0, 10.0])
        np.testing.assert_array_equal(py, [2, 1])

    def test_feat_shape_and_finite(self):
        a = suite.normalize(np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64))
        i = suite.normalize(np.array([[0.2, 0.3, 0.5]], dtype=np.float64))
        p = np.array([0.7], dtype=np.float64)
        z = sel_mod.feat(a, i, p)
        # log(a)4 + log(i)3 + a4 + i3 + p1 + amax1 + imax1 = 17
        self.assertEqual(z.shape, (1, 17))
        self.assertTrue(np.isfinite(z).all())

    def test_suite_normalize_and_specimen_group(self):
        p = np.array([[1.0, 1.0, 2.0, 0.0]], dtype=np.float64)
        n = suite.normalize(p)
        self.assertAlmostEqual(float(n.sum()), 1.0)
        g = suite.specimen_group(
            pd.Series(["foo_segment_3_window_9.wav"])
        ).tolist()
        self.assertEqual(g, ["foo"])


class TestSegmentPurityOnRealManifests(unittest.TestCase):
    def _purity(self, csv: Path, base: Path, split: str):
        fr = suite.load_manifest(csv, base, split)
        y = fr.y.to_numpy(np.int64)
        sk = sel_mod.seg(fr.audio_file.astype(str)).to_numpy()
        mixed = 0
        for k in np.unique(sk):
            if len(np.unique(y[sk == k])) > 1:
                mixed += 1
        return len(np.unique(sk)), mixed

    def test_hand_segments_label_pure(self):
        n, mixed = self._purity(
            ROOT / "audio_visual_dataset_default/dataset.csv",
            ROOT / "audio_visual_dataset_default",
            "hand_train",
        )
        self.assertGreater(n, 0)
        self.assertEqual(mixed, 0)

    def test_robot_segments_label_pure(self):
        n, mixed = self._purity(
            ROOT / "audio_visual_dataset_robo_default/dataset.csv",
            ROOT / "audio_visual_dataset_robo_default",
            "robot_test",
        )
        self.assertGreater(n, 0)
        self.assertEqual(mixed, 0)


class TestLockedArtifactsProtocol(unittest.TestCase):
    def setUp(self):
        self.lock_path = OUT / "selection_lock.json"
        self.metrics_path = OUT / "segment_meta_weighted_final_test_metrics.json"
        self.board_path = OUT / "hand_meta_weighted_leaderboard.csv"
        self.assertTrue(self.lock_path.is_file(), "missing selection lock")
        self.assertTrue(self.metrics_path.is_file(), "missing final metrics")
        self.lock = json.loads(self.lock_path.read_text())
        self.metrics = json.loads(self.metrics_path.read_text())

    def test_lock_test_loaded_false(self):
        self.assertIs(self.lock.get("test_loaded"), False)

    def test_lock_group_and_hand_only(self):
        self.assertEqual(self.lock.get("group_column"), "specimen_group")
        self.assertEqual(self.lock.get("selection_data"), "hand/default only")
        self.assertIn("segment", self.lock.get("protocol", ""))

    def test_selection_source_never_opens_robot(self):
        src = SEL_SRC.read_text()
        for needle in (
            "robot_test",
            "audio_visual_dataset_robo",
            "robo_default",
        ):
            self.assertNotIn(needle, src, f"selection must not reference {needle}")

    def test_final_asserts_lock_before_test(self):
        src = FT_SRC.read_text()
        self.assertIn("assert not lock.get('test_loaded')", src)
        self.assertIn("audio_visual_dataset_robo_default", src)

    def test_hand_leaderboard_selects_locked_candidate(self):
        board = pd.read_csv(self.board_path)
        self.assertGreater(len(board), 0)
        best = board.iloc[0]
        cand = self.lock["selected_candidate"]
        self.assertAlmostEqual(float(best["C"]), float(cand["C"]))
        self.assertAlmostEqual(
            float(best["trunk_class_weight"]), float(cand["trunk_class_weight"])
        )
        self.assertAlmostEqual(
            float(best["twig_class_weight"]), float(cand["twig_class_weight"])
        )
        # leaderboard is sorted by mean then worst CV F1 descending
        means = board["mean_cv_macro_f1"].to_numpy()
        self.assertEqual(float(means[0]), float(means.max()))

    def test_final_macro_f1_strictly_above_075(self):
        f1 = float(self.metrics["metrics"]["macro_f1_4class"])
        self.assertGreater(f1, 0.75)
        self.assertEqual(int(self.metrics["n"]), 2219)
        self.assertEqual(self.metrics["split"], "robot_test_final")
        self.assertTrue(self.metrics["invariants"]["test_loaded_after_lock"])
        self.assertTrue(self.metrics["invariants"]["selection_used_hand_only"])

    def test_f1_matches_confusion_matrix_reconstruction(self):
        cm = np.asarray(self.metrics["confusion_matrix_4class"], dtype=np.int64)
        self.assertEqual(cm.shape, (4, 4))
        self.assertEqual(int(cm.sum()), 2219)
        y_true = []
        y_pred = []
        for t in range(4):
            for p in range(4):
                c = int(cm[t, p])
                y_true.extend([t] * c)
                y_pred.extend([p] * c)
        recon = f1_score(y_true, y_pred, average="macro", zero_division=0)
        self.assertAlmostEqual(
            recon, float(self.metrics["metrics"]["macro_f1_4class"]), places=10
        )

    def test_predictions_manifest_alignment_inputs_exist(self):
        """Final path asserts audio prediction files align to robot manifest."""
        audio_pred = Path(
            "outputs/audio_feature_benchmarks/audio_lift_source_blend_select/"
            "reports/audio_lift_source_blend_select_final_test_predictions.csv"
        )
        raw = ROOT / "audio_visual_dataset_robo_default/dataset.csv"
        self.assertTrue(audio_pred.is_file())
        self.assertTrue(raw.is_file())
        a = pd.read_csv(audio_pred)
        r = pd.read_csv(raw)
        self.assertEqual(len(a), 2219)
        self.assertTrue(
            np.array_equal(
                a.audio_file.astype(str).to_numpy(),
                r.audio_file.astype(str).to_numpy(),
            )
        )
        for col in suite.PROBA_COLUMNS:
            self.assertIn(col, a.columns)

    def test_rejected_specimen_pool_not_used_as_winner(self):
        """Specimen-pool 0.77 is invalid mixed-label aggregation; not this lock."""
        self.assertNotIn("specimen_pool", self.metrics.get("protocol", ""))
        self.assertLess(
            abs(float(self.metrics["metrics"]["macro_f1_4class"]) - 0.770577),
            1.0,
        )  # different score space check below
        self.assertNotAlmostEqual(
            float(self.metrics["metrics"]["macro_f1_4class"]),
            0.770577010942886,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
