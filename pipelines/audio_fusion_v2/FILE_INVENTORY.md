# File inventory — source-only audio fusion v2

This package is deliberately **not self-contained**. It contains code and
code-side configuration only; the generated `CODE_ONLY_MANIFEST.txt` is the
authoritative archive listing.

## Included code

| Path | Role |
|---|---|
| `e2e_code/run_segment_meta_weighted_group_selection.py` | Generates hand-only `hand_meta_oof_features.npy` and `hand_meta_oof_y.npy` |
| `e2e_code/run_segment_rule_stack_v2_group_selection.py` | Hand-only group-OOF search and v2 lock |
| `e2e_code/run_segment_rule_stack_v2_group_final_test.py` | Locked hand fit, single robot/test run, and v2-base output writer |
| `e2e_code/` | Feature extraction, OOF, selection, final-test, and claim runners |
| `audio_only/code/train_chain/` | Audio-chain sources consumed by v2 OOF stages |
| `fusion_v2/code/` | Fusion evaluator and paper-claim implementation |
| `shared/metrics_utils.py` | Shared metric helpers |

## Excluded by design

- Raw WAV/JPEG media
- All extracted features (`.npy`, `.npz`)
- Dataset manifests, predictions, metrics, locks, and other run artifacts (`.csv`, result JSON)
- `outputs/` caches and external symlinks

The archive builder aborts if it detects `.npy`, `.npz`, or `.csv` in the
staging tree.
