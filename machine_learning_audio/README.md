# Audio-Only Paper-Safe Current Checkpoint

Checkpoint date: 2026-07-06

This checkpoint freezes the current best audio-only classical-ML protocol before any further experiments.

## Main Result

- Run slug: `audio_lift_source_blend_select`
- Final split: `robot_test_final`
- Final macro F1, 4-class: `0.7026719927789447`
- Final accuracy, 4-class: `0.7967552951780081`
- Final contact macro F1: `0.625050260344378`
- Final binary macro F1: `0.9291164642187257`

## Locked Selection

- Selection data: hand/default train labels plus locked audio-only OOF probabilities
- Final robot/test labels used for selection: no
- Image or multimodal features: no
- Deep learning: no
- Unsupervised adaptation on test: no
- Selected source: `report_gate_onehot`
- Selected source weight: `0.05`
- Selected blend mode: `segment_lift`
- Selected OOF macro F1: `0.9353447727801658`
- Selection score: `0.9352330939963467`

The selection lock is stored in:

- `audio_lift_source_blend_select_selected_without_test.json`
- `audio_lift_source_blend_select_method_card_before_test.json`

## Paper Note

This checkpoint is suitable to report as an audio-only classical-ML pipeline selected by hand/train OOF validation, with robot/test used only for final evaluation inside the locked script.

Important nuance: the final protocol uses segment/specimen-level consistency at inference through filename-derived group equality. It does not parse labels from the test set, but it should be described as group-consistency inference rather than strict independent-window inference.

## Snapshot Files

- `train_audio_lift_source_blend_select_final_test.py`
- `train_audio_report_grade_gate_select_final_test.py`
- `train_audio_log_consensus_pair_blend_select_final_test.py`
- `AUDIO_ONLY_REPORT_GRADE_PROTOCOL.md`
- `audio_lift_source_blend_select_selected_without_test.json`
- `audio_lift_source_blend_select_method_card_before_test.json`
- `audio_lift_source_blend_select_protocol_summary.json`
- `audio_lift_source_blend_select_final_test_report.csv`
- `audio_lift_source_blend_select_final_test_confusion_matrix.csv`
- `audio_lift_source_blend_select_oof_leaderboard.csv`
- `SHA256SUMS.txt`

Original artifact directory:

`outputs/audio_feature_benchmarks/audio_lift_source_blend_select/`
