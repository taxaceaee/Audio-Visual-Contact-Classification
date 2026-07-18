# Multimodal fusion experiment audit

## Non-negotiable protocol

- Split unit: `specimen_group`, obtained by removing `_segment_...` from the
  audio stem. This keeps every crop from one long source video in one fold.
- Selection: hand/default only, five-fold `StratifiedGroupKFold`.
- Test: robot-only is opened only after a selection lock and evaluated once per
  locked candidate.
- No filename-derived class feature is used.
- For the audio-only-compatible branches, the exact transform is
  `anchor_lift_proba(0.8 * highsr + 0.2 * pairwise)`, including the audio
  pipeline's segment/specimen contact lift.

## Results

| branch | hand selection result | robot/test Macro F1 | status |
|---|---:|---:|---|
| old stress late fusion | worst grouped view 0.8607 | 0.7132 | historical result; source mismatch between selection and final artifact, not paper-safe |
| exact anchor + CLIP, log | mean grouped 0.9594 | 0.6861 | valid, no class bias |
| exact anchor + CLIP, grouped bias | mean grouped 0.9620 | 0.7033 | valid; best exact-anchor bias branch |
| audio-only locked blend + CLIP | mean grouped 0.9496 | 0.7091 | valid; current best reproducible exact branch |
| audio-only locked blend + CLIP + grouped bias | mean grouped 0.9526 | 0.6776 | valid, rejected on robot after lock |
| spatial image + exact anchor | mean grouped 0.9316 | 0.7004 | valid |
| contact-subclass total240 + EfficientNet | mean grouped 0.8539 | 0.6007 | valid |
| HGB on CLIP + exact anchor | mean grouped 0.9762 | 0.6118 | valid; severe hand→robot shift |
| exact anchor + CLIP KNN | mean grouped 0.9432 | 0.5897 | valid |
| exact anchor + CLIP MLP | mean grouped 0.8704 | not opened | rejected on hand selection |

Current best valid final metrics are in
`outputs/audio_feature_benchmarks/audio_only_locked_plus_clip_group_selection/`
and have Macro F1 `0.7090736`, accuracy `0.7976566`, and binary Macro F1
`0.9277353`. Its per-class F1 is ambient `0.9344`, leaf `0.7706`, trunk
`0.5431`, twig `0.5882`.

## Diagnosis

The hand grouped validation scores are consistently much higher than robot
test scores across representations and meta-models. This is a domain shift
from the hand/default recordings and images to the robot recordings, not a
simple late-fusion-weight problem. The persistent bottleneck is trunk recall;
ambient/noambient separation remains strong.

The next valid direction is domain-aware representation/training using only
hand data: stronger robot-like image/audio augmentations, frozen-feature
domain-robust training, and selection on grouped stress/domain folds. Any
unlabeled robot adaptation or filename prior must remain a separately labeled
non-strict diagnostic and cannot be used for the strict result.

