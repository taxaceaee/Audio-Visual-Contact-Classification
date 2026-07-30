# Methods README — Audio-Only Specimen Consensus for Tree-Structure Contact Classification

## Abstract-Style Summary

This document describes the current paper-ready audio-only pipeline for contact classification in agricultural tree structures. The method classifies audio samples into four classes: `ambient`, `leaf`, `trunk`, and `twig`. Unlike the earlier audio-visual direction of the project, the current pipeline does not use images, visual descriptors, multimodal fusion, or deep-learning feature extraction at final inference. Instead, it uses a locked audio-only classical machine-learning checkpoint with out-of-fold model selection, group-consistency inference, segment/specimen aggregation, and a contact-mass lifting mechanism.

The validated checkpoint is `audio_lift_source_blend_select`. Under the frozen artifact replay protocol, it achieves `0.7967552951780081` four-class accuracy, `0.7026719927789447` four-class macro-F1, `0.625050260344378` contact macro-F1, and `0.9291164642187257` binary ambient/contact macro-F1 on the final `robot_test_final` split with `2,219` samples.

## 1. Problem Formulation

The task is contact classification for robotic interaction with agricultural tree structures. Given an audio recording or audio window produced during interaction with tree material, the model predicts one of four labels:

| Label | Meaning | Contact Type |
| --- | --- | --- |
| `ambient` | background or non-contact audio | non-contact |
| `leaf` | contact with foliage | contact |
| `trunk` | contact with trunk/large woody structure | contact |
| `twig` | contact with small branch/twig | contact |

For evaluation, two formulations are used:

1. **Four-class classification**: `ambient`, `leaf`, `trunk`, `twig`.
2. **Binary contact classification**: `ambient` vs contact, where contact merges `leaf`, `trunk`, and `twig`.

The binary view is useful because robotic action often first needs to determine whether meaningful contact occurred. The four-class view is harder but more informative because it separates different contact materials.

## 2. Motivation

The original project reference paper, `Audio-Visual Contact Classification for Tree Structures in Agriculture`, motivates the use of contact sensing for agricultural robotics. That direction used audio-visual information. The current method deliberately shifts toward an **audio-only** pipeline.

This shift is motivated by deployment concerns:

- cameras may be unreliable in occluded foliage;
- lighting and viewpoint can vary strongly in orchards;
- audio sensors are lightweight and inexpensive;
- audio-only inference reduces dependency on visual preprocessing;
- contact audio provides direct evidence of physical interaction.

However, audio-only classification is difficult because contact sounds are short, noisy, and imbalanced. The `ambient` class is often easier and more frequent, while `leaf`, `trunk`, and `twig` can be confused with each other.

## 3. Dataset and Split Protocol

The project dataset is stored under the repository dataset tree, with notebook-compatible paths mapped to the audio-visual dataset folders. The master validation notebook uses the frozen checkpoint artifacts in `machine_learning_audio`.

The validated final split is:

```text
robot_test_final
```

The final test set contains:

```text
2,219 samples
```

The checkpoint documentation also records the training-side selection data as hand/default train labels plus locked audio-only out-of-fold probabilities. The selection procedure does not use robot/test labels before final evaluation.

## 4. Audio-Only Constraint

The current paper-safe checkpoint explicitly follows these constraints:

Allowed:

- hand/default train labels;
- locked audio-only out-of-fold probabilities;
- filename-derived equality/grouping keys for windows belonging to the same segment;
- filename-derived equality/grouping keys for segments belonging to the same specimen;
- final robot/test audio predictions only after selection lock.

Forbidden:

- image features;
- multimodal features;
- visual descriptors;
- robot/test labels during model selection;
- robot/test probabilities during model selection;
- parsing class words such as `leaf`, `trunk`, `twig`, `ambient`, or `contact` from filenames as predictions;
- unsupervised adaptation on robot/test data.

This distinction matters for paper writing. The method uses group consistency from filename structure, but it does not use label leakage from filenames.

## 5. Reproducibility Modes

Two reproduction modes exist.

### 5.1 Mode A — Frozen Artifact Replay

Mode A is the recommended mode for paper validation. It reuses the locked checkpoint artifacts and recomputes the final report from frozen outputs. This mode is designed to reproduce the exact reported metrics.

Used files include:

- `audio_lift_source_blend_select_final_test_report.csv`
- `audio_lift_source_blend_select_final_test_confusion_matrix.csv`
- `audio_lift_source_blend_select_oof_leaderboard.csv`
- `audio_lift_source_blend_select_selected_without_test.json`
- `audio_lift_source_blend_select_method_card_before_test.json`
- `audio_lift_source_blend_select_protocol_summary.json`

The master validation notebook created for this repository is:

```text
machine_learning_audio/00_master_audio_paper_pipeline.ipynb
```

The executed notebook is:

```text
machine_learning_audio/00_master_audio_paper_pipeline.executed.ipynb
```

Its validation summary is:

```text
machine_learning_audio/00_master_audio_paper_pipeline_validation_summary.json
```

### 5.2 Mode B — Full Rebuild From Raw Audio

Mode B rebuilds the pipeline from raw audio by rerunning feature extraction, cross-validation, source models, blending, selection, and final evaluation. This mode is more complete but can drift slightly across machines because of package versions, random seeds, and multithreaded numerical behavior.

Mode B is useful for development but should not be used as the only source of final paper numbers unless the full environment is frozen.

## 6. Pipeline Overview

The current paper-ready method can be summarized as:

```text
Raw audio / audio windows
        ↓
Audio feature and source-model predictions
        ↓
Train-only OOF source selection
        ↓
Selection lock before final test access
        ↓
Final robot/test probability loading
        ↓
Segment-level grouping
        ↓
Specimen-level contact consensus
        ↓
Contact-mass lift
        ↓
Four-class prediction
        ↓
Final metrics and paper artifacts
```

The key methodological contribution is not a new neural architecture. The contribution is a robust audio-only decision pipeline that combines locked OOF selection with segment/specimen-level consistency.

## 7. Source Models and Feature Families

The checkpoint uses selected audio-only prediction sources. The broader `machine_learning_audio` folder contains scripts for several source-model families, including:

- high-sample-rate temporal test-time augmentation;
- pairwise contact/stress cross-validation;
- report-grade gating;
- source blending;
- specimen contact consensus;
- specimen contact lift;
- OOF stacking;
- multifeature TTA grid ensembles.

Important scripts include:

- `train_audio_highsr_temporal_tta_select_final_test.py`
- `train_audio_pairwise_contact_stress_cv_select_final_test.py`
- `train_audio_lift_source_blend_select_final_test.py`
- `train_audio_report_grade_gate_select_final_test.py`
- `train_audio_specimen_contact_consensus_select_final_test.py`
- `train_audio_specimen_contact_lift_select_final_test.py`

The final validated checkpoint is:

```text
audio_lift_source_blend_select
```

The selected source in the lock file is:

```text
report_gate_onehot
```

with source weight:

```text
0.05
```

and blend mode:

```text
segment_lift
```

## 8. Locked OOF Selection

Model/source selection is performed on train-only out-of-fold predictions. This is a central anti-leakage design choice.

The selection rule is described as:

```text
highest train-only OOF macro/contact score over lift-anchor source blends
```

The selected configuration has:

| Quantity | Value |
| --- | ---: |
| selected source | `report_gate_onehot` |
| source weight | `0.05` |
| blend mode | `segment_lift` |
| selected OOF macro-F1 | `0.9353447727801658` |
| selected OOF contact macro-F1 | `0.9164944534943943` |
| selected OOF binary macro-F1 | `0.99077894279929` |
| selection score | `0.9352330939963467` |

The selection lock is written before final test evaluation. This should be described explicitly in the paper because it protects the final test split from tuning leakage.

## 9. Segment-Level Group Consistency

Audio recordings are split into windows or segments. Independent window-level classification can be unstable because each short window may contain partial, noisy, or ambiguous evidence.

The pipeline derives a `group_key` from filename structure to identify windows belonging to the same audio segment. Segment-level aggregation improves stability by combining predictions from related windows.

This mechanism is allowed because it uses equality/grouping structure, not class information.

## 10. Specimen-Level Consensus

Beyond segment grouping, the method also uses specimen-level grouping. A specimen represents a physical object or repeated contact unit inferred from filename structure.

The core idea is:

- multiple segments from the same specimen should often share compatible contact behavior;
- if several segments indicate a likely contact class, that consensus can stabilize uncertain predictions;
- specimen-level aggregation is especially useful when single-window predictions overpredict `ambient`.

Let each segment prediction be a probability vector:

```text
p_i = [p_i(ambient), p_i(leaf), p_i(trunk), p_i(twig)]
```

The contact part is:

```text
p_i(contact) = [p_i(leaf), p_i(trunk), p_i(twig)]
```

For a specimen `s`, contact-likely segments are selected using a threshold. The notebook configuration uses:

```text
SPECIMEN_THRESHOLD = 0.45
```

The specimen consensus distribution can be described as:

```text
c_s = normalize(mean_i p_i(contact))
```

where the mean is taken over contact-likely segments for the same specimen.

## 11. Contact-Mass Lift

Contact-mass lift is designed to correct a common failure mode: segments that are predicted as `ambient` even though their specimen has strong contact evidence.

The final notebook configuration uses:

| Parameter | Value |
| --- | ---: |
| `MIN_CONTACT_SEGMENTS` | `1` |
| `LIFT_MIN_MASS` | `0.35` |
| `LIFT_FLOOR` | `0.58` |
| `LIFT_CONFIDENCE` | `0.45` |

At a high level:

1. The method estimates contact mass for each segment.
2. It computes a specimen-level contact consensus over `leaf`, `trunk`, and `twig`.
3. If the specimen consensus is confident enough, likely contact segments can be lifted away from `ambient`.
4. Lifted contact probability is distributed according to the specimen consensus.

This makes the final classifier less vulnerable to local ambient overprediction while preserving a strict audio-only inference pipeline.

## 12. Final Evaluation Metrics

The pipeline reports:

- four-class accuracy;
- four-class macro-F1;
- four-class macro precision;
- four-class macro recall;
- weighted F1;
- contact macro-F1 over `leaf`, `trunk`, and `twig`;
- binary ambient/contact macro-F1;
- per-class precision, recall, F1, and support;
- confusion matrix.

Macro-F1 is especially important because the dataset is class-imbalanced. Accuracy alone can overstate performance when `ambient` dominates.

## 13. Final Validated Results

Final split:

```text
robot_test_final
```

Number of samples:

```text
2,219
```

Main metrics:

| Metric | Value |
| --- | ---: |
| four-class accuracy | `0.7967552951780081` |
| four-class macro-F1 | `0.7026719927789447` |
| contact macro-F1 | `0.625050260344378` |
| binary ambient/contact macro-F1 | `0.9291164642187257` |

Per-class final metrics:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `ambient` | `0.8788819875776398` | `1.0` | `0.9355371900826446` | `1132` |
| `leaf` | `0.6579572446555819` | `0.9453924914675768` | `0.7759103641456583` | `293` |
| `trunk` | `0.9761904761904762` | `0.3557483731019523` | `0.5214626391096979` | `461` |
| `twig` | `0.5701754385964912` | `0.5855855855855856` | `0.5777777777777777` | `333` |

Confusion matrix:

| true \ pred | ambient | leaf | trunk | twig |
| --- | ---: | ---: | ---: | ---: |
| ambient | 1132 | 0 | 0 | 0 |
| leaf | 2 | 277 | 0 | 14 |
| trunk | 126 | 38 | 164 | 133 |
| twig | 28 | 106 | 4 | 195 |

## 14. Result Interpretation

The final result shows strong separation between `ambient` and contact, reflected in the binary macro-F1 of approximately `0.9291`. The four-class macro-F1 is lower because distinguishing contact materials remains difficult.

Important observations:

- `ambient` recall is perfect on the final confusion matrix.
- `leaf` recall is high, but precision is lower because some `twig` and `trunk` cases are predicted as `leaf`.
- `trunk` precision is very high, but recall is low. This means the model is conservative when predicting `trunk`.
- `twig` remains challenging, with confusion against `leaf` and `trunk`.

This supports a nuanced paper claim: the system is effective at audio-only contact detection and achieves useful four-class classification, but material-level contact discrimination still has room for improvement.

## 15. Safe Paper Claims

The following claims are supported by current evidence:

- The method is audio-only.
- The final checkpoint does not use image or multimodal features.
- Model/source selection is locked before final robot/test evaluation.
- The final `robot_test_final` evaluation contains `2,219` samples.
- The validated checkpoint achieves `0.7967552951780081` accuracy and `0.7026719927789447` macro-F1.
- Group-consistency inference and contact-mass lift improve the paper framing by using repeated segment/specimen structure.
- Binary contact detection is substantially stronger than fine-grained contact-material classification.

## 16. Claims to Avoid Unless More Experiments Are Added

Avoid these claims unless additional evidence is produced:

- global state-of-the-art across all audio-visual contact classification systems;
- superiority over every deep audio foundation model;
- robustness across all orchards, devices, seasons, or crop types;
- full independence of each test window, because group consistency is used at inference;
- complete end-to-end training reproducibility from raw audio under every environment.

Better wording:

```text
The proposed audio-only pipeline provides a paper-safe, locked-evaluation checkpoint and demonstrates that specimen/group consistency can improve robust contact classification under the current robot-test protocol.
```

## 17. Limitations and Threats to Validity

The main limitations are:

- the final validation uses frozen artifacts rather than a full raw-audio rebuild;
- the method relies on filename-derived grouping equality;
- material-level confusion remains significant, especially for `trunk` and `twig`;
- full reproducibility may vary across package versions in Mode B;
- external validation on additional orchards or robot platforms is not yet included;
- claims against deep audio foundation models require same-protocol experiments.

These limitations should be stated transparently in the paper.

## 18. Recommended Paper Framing

Recommended title:

```text
Specimen-Consensus Audio Models for Contact Classification in Agricultural Tree Structures
```

Recommended contribution statement:

```text
We propose an audio-only, paper-safe contact-classification pipeline that combines train-only OOF source selection, group-consistency inference, specimen-level contact consensus, and contact-mass lift for robust robot-test evaluation.
```

Recommended result statement:

```text
Under the frozen artifact replay protocol, the selected audio-only checkpoint achieves 0.7968 accuracy, 0.7027 four-class macro-F1, 0.6251 contact macro-F1, and 0.9291 binary contact macro-F1 on 2,219 robot-test samples.
```

## 19. Files to Cite Internally

Primary validation notebook:

```text
machine_learning_audio/00_master_audio_paper_pipeline.executed.ipynb
```

Validation summary:

```text
machine_learning_audio/00_master_audio_paper_pipeline_validation_summary.json
```

Final report:

```text
machine_learning_audio/audio_lift_source_blend_select_final_test_report.csv
```

Final confusion matrix:

```text
machine_learning_audio/audio_lift_source_blend_select_final_test_confusion_matrix.csv
```

Selection lock:

```text
machine_learning_audio/audio_lift_source_blend_select_selected_without_test.json
```

Method card:

```text
machine_learning_audio/audio_lift_source_blend_select_method_card_before_test.json
```

Full guide:

```text
machine_learning_audio/AUDIO_ONLY_FULL_REIMPLEMENTATION_GUIDE.md
```

## 20. Short Version for Paper Methods Section

We use an audio-only contact-classification pipeline for agricultural tree structures. The system predicts four classes: ambient, leaf, trunk, and twig. Candidate audio sources are selected using train-only out-of-fold predictions, and the selected configuration is locked before robot-test evaluation. During inference, predictions are aggregated using segment-level and specimen-level grouping derived from filename equality rather than class labels. A specimen consensus distribution is estimated over contact classes, and contact-mass lift is applied to reduce ambient overprediction when specimen-level contact evidence is confident. The final checkpoint, `audio_lift_source_blend_select`, is evaluated on the `robot_test_final` split with 2,219 samples and achieves 0.7968 accuracy, 0.7027 macro-F1, 0.6251 contact macro-F1, and 0.9291 binary contact macro-F1.
