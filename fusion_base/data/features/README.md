# Precomputed features (not tracked in git)

Place sealed feature blobs here for `reproduce.py` (path: `fusion_base/data/features/`):

```text
features/
  clip/hand_train_full/X.npy
  clip/robot_test/X.npy
  total240/hand_train_full/X.npy
  total240/robot_mix/X.npy
  total240/robot_test/X.npy
  v2_base/segment_rule_stack_v2_base_segment_outputs.npz
  wav2vec2/hand_train_full_X.npy
  wav2vec2/robot_test_X.npy
```

These files are gitignored (`*.npy`, `*.npz`). Keep them locally or restore from the portable archive.
