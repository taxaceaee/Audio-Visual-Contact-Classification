# Image-only handcrafted ML attempt log

Goal: use only image-derived handcrafted features plus classical ML, split hand/default
into train/val, keep robot/test as final holdout, avoid audio/multimodal features,
and target robot/test 4-class macro F1 > 0.5.

## Data protocol

- Train source: `audio_visual_dataset_default/dataset.csv` from `/home/ttung05/Desktop/tree_base/tree_structures`
- Final test source: `audio_visual_dataset_robo_default/dataset.csv`
- Labels: `ambient`, `leaf`, `trunk`, `twig`
- Audio files are carried only as manifest identifiers in reports; no audio features are used.
- Current image feature caches:
  - v1 handcrafted cache: `outputs/audio_feature_benchmarks/multimodal_classical_cv_select/image_features`
  - spatial-v2 cache: `outputs/audio_feature_benchmarks/image_spatial_v2_features`

## Runs completed

| Run | Selection split | Selection result | Robot/test macro F1 | Notes |
| --- | --- | ---: | ---: | --- |
| `image_handcrafted_ml_select` | segment grouped | val macro F1 0.586971 | 0.318536 | v1 compact logistic with tuned bias; val was optimistic. |
| `image_segment_consensus_select` | segment grouped | val macro F1 0.586971 | 0.318536 | Segment consensus did not change validation predictions. |
| `image_segment_consensus_specimen_select` | specimen grouped | val macro F1 0.504534 | 0.300875 | Harder split reduced leakage optimism; selected two-model ensemble. |
| `image_spatial_v2_specimen_select` | specimen grouped | regularized val score 0.426376 | 0.301585 | Added spatial thumbnail/grid/crop features; improved accuracy but not macro F1. |
| `image_hierarchical_ml_specimen_select` | specimen grouped | regularized val score 0.493858 | 0.236181 | Ambient/contact + contact subtype hierarchy; regularized selection chose no bias, transferred poorly. |
| `image_hierarchical_ml_macro_select` | specimen grouped | val macro F1 0.527618 | 0.254472 | Same hierarchy with unregularized macro-F1 selection; tuned bias overfit hand validation. |

## Current best robot/test result

- Best 4-class robot/test macro F1: **0.318536**
- Best runs:
  - `outputs/audio_feature_benchmarks/image_handcrafted_ml_select/reports/image_handcrafted_ml_select_final_test_report.csv`
  - `outputs/audio_feature_benchmarks/image_segment_consensus_select/reports/image_segment_consensus_select_final_test_report.csv`

## Findings so far

- Segment-level train/val split leaks specimen identity across train and val; specimen split is more honest and much harder.
- Segment consensus does not help because each segment is already label-pure and candidate predictions are mostly unchanged after bias.
- Large class-bias tuning can raise hand validation macro F1 but does not transfer to robot/test.
- Spatial-v2 features raise robot/test accuracy to 0.441190 and binary macro F1 to 0.537627, but 4-class macro F1 remains low because contact-class separation is weak.

## Next train-only directions

- Build image stress views from hand train only: grayscale, color equalization, blur, crop, brightness/contrast, JPEG-like degradation.
- Select candidates by specimen-CV worst-case macro F1 across clean and stress views before any robot/test load.
- Prefer low-bias or bias-regularized selection because high validation bias has not transferred.
- If strict holdout purity is required, stop using the current robot/test for iterative feedback and reserve a fresh robot holdout for any final claim.
