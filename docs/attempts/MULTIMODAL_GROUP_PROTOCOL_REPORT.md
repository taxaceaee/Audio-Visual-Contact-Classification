# Multimodal images + audio: grouped evaluation report

## Protocol locked

The split follows the audio-only pipeline's provenance rule. Each window is
first traced back to its source specimen/video:

```text
segment_group = remove `_window_<number>...` from the audio stem
specimen_group = remove `_segment_...` from the audio stem
```

`specimen_group` is the split unit. All windows, images, and audio crops from
one source specimen stay in exactly one hand split. The selected split has:

- hand: 10,676 rows, 235 specimen groups;
- train: 8,292 rows, 188 groups;
- validation: 2,384 rows, 47 groups;
- train/validation specimen-group overlap: 0;
- hand/test specimen overlap: 0 and segment overlap: 0.

The final candidate was selected only from hand data using five-fold
`StratifiedGroupKFold`, with stress views `clean`, `robot_mix`, and `bandlimit`.
The robot/test manifest is not loaded by the selection script. Fusion is
selected once, then evaluated once on `audio_visual_dataset_robo_default`.

Specimen grouping is used for splitting, not for forcing one class over a
whole specimen. This is necessary because 202/235 hand specimens and 44/45
robot specimens contain multiple labels. Segment-level aggregation is kept
for inference.

## Locked fusion

```text
image: ResNet18 embedding -> balanced logistic regression, C=0.1
audio: locked audio-only high-SR temporal candidate + pairwise source blend
fusion: log-probability fusion
audio weight: 0.65
image weight: 0.35
```

Five-fold hand-only selection:

| view | mean Macro F1 | worst fold Macro F1 |
|---|---:|---:|
| clean | 0.932970 | 0.909218 |
| robot_mix | 0.927490 | 0.894139 |
| bandlimit | 0.909544 | 0.860737 |
| mean across views | 0.923335 | 0.860737 worst view/fold |

## Final robot-only test

This is the first result after the lock; it must not be used to choose the
fusion weight.

| metric | value |
|---|---:|
| Accuracy, 4 class | 0.801262 |
| Macro precision, 4 class | 0.772389 |
| Macro recall, 4 class | 0.724306 |
| Macro F1, 4 class | **0.713231** |
| Weighted F1, 4 class | 0.782064 |
| Binary accuracy ambient/noambient | 0.924741 |
| Binary macro precision | 0.935720 |
| Binary macro recall | 0.923183 |
| Binary macro F1 | 0.924048 |

Per-class test metrics:

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| ambient | 0.871440 | 1.000000 | 0.931304 | 1,132 |
| leaf | 0.648910 | 0.914676 | 0.759207 | 293 |
| trunk | 0.948187 | 0.396963 | 0.559633 | 461 |
| twig | 0.621019 | 0.585586 | 0.602782 | 333 |

Confusion matrix, rows=true and columns=predicted:

```text
             ambient  leaf  trunk  twig
ambient          1132     0      0     0
leaf                5   268      6    14
trunk             132    41    183   105
twig               30   104      4   195
```

## Interpretation

The target Macro F1 0.80 has not been reached. The main bottleneck is not
ambient detection: binary Macro F1 is 0.924. It is the contact subclass
boundary, especially trunk recall (0.397) and twig/leaf confusion. The hand
group validation is substantially easier than the robot domain, so its high
score cannot be treated as evidence that the test score will be >=0.80.

The attempted frozen HuBERT branch was not included in the result: the model
checkpoint download was too slow and was stopped before feature extraction.
No test result was obtained from that branch.

## Reproducible artifacts

- `run_multimodal_multifold_group_selection.py`
- `outputs/audio_feature_benchmarks/multimodal_multifold_group_selection/selection_lock.json`
- `outputs/audio_feature_benchmarks/multimodal_multifold_group_selection/hand_multifold_group_leaderboard.csv`
- `run_multimodal_stress_robust_group_final_test.py`
- `outputs/audio_feature_benchmarks/multimodal_stress_robust_group_selection/stress_robust_final_test_metrics.json`

