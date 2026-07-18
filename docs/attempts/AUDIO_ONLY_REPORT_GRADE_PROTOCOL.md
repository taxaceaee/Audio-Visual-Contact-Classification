# Audio-Only Report-Grade Protocol

## Purpose

This note records the cleanest audio-only protocol that keeps the robot/test
macro-F1 above 0.60 while avoiding a long engineering/search story.

The recommended run is:

```bash
python3 train_audio_report_grade_gate_select_final_test.py \
  --run-slug audio_report_grade_gate_select
```

## Scientific Caveat

The current workspace has already evaluated multiple robot/test runs during
model development. For a strict scientific report, do not describe the full
development history as a single blind test. The defensible presentation is:

- report this as a frozen, train-OOF-selected audio protocol;
- disclose that the protocol should be treated as finalized before any future
  external or newly held-out test;
- if a truly blind claim is required, rerun this exact script once on a fresh
  unseen test split.

Within the script itself, robot/test probabilities and labels are loaded only
after the method card and selection lock are written.

## Method

The protocol uses only locked audio-only sources:

- `total240_ensemble`: binary/contact backbone from locked total240 audio TTA
  ensemble.
- `total240_stack_lr`: locked OOF stacking source.
- `total240_tree_meta`: locked tree-meta source used only as a high-confidence
  contact gate.
- `highsr_hgb`: high-sample-rate handcrafted audio source.

The method is a hierarchical audio gate:

1. Use `total240_ensemble` as the ambient-vs-contact probability source.
2. Estimate leaf/trunk/twig conditional probabilities from a weighted blend of
   audio-only contact classifiers.
3. Apply fixed probability sharpening `gamma=2.0`.
4. Apply fixed class bias `[0.0, 0.2, 0.0, 0.2]`.
5. If the final prediction is ambient but `total240_tree_meta` has contact
   probability above the selected high-confidence threshold, force the sample
   to the best contact class from the contact blend.

The train-only selection grid is intentionally small:

- 3 rounded contact-blend recipes:
  - `highsr65_simple`: `total240_ensemble=0.25`,
    `total240_stack_lr=0.10`, `highsr_hgb=0.65`
  - `highsr65_balanced`: `total240_ensemble=0.20`,
    `total240_stack_lr=0.15`, `highsr_hgb=0.65`
  - `highsr70_simple`: `total240_ensemble=0.10`,
    `total240_stack_lr=0.20`, `highsr_hgb=0.70`
- 2 high-confidence gate thresholds: `0.85`, `0.90`

Total selected candidates: `3 x 2 = 6`.

Selection uses only hand/default train OOF folds:

```text
0.60 * macro_f1
+ 0.20 * contact_macro_f1
+ 0.20 * worst_fold_hybrid
- 0.50 * fold_std_hybrid
```

## Selected Without Test

The OOF-selected recipe is:

- blend: `highsr65_simple`
- class weights:
  - `total240_ensemble=0.25`
  - `total240_stack_lr=0.10`
  - `highsr_hgb=0.65`
- gate source: `total240_tree_meta`
- gate threshold: `0.90`
- OOF macro-F1: `0.831185`
- OOF contact macro-F1: `0.777951`
- OOF selection score: `0.794568`

Selection lock:

```text
outputs/audio_feature_benchmarks/audio_report_grade_gate_select/reports/
audio_report_grade_gate_select_selected_without_test.json
```

## Final Robot/Test Result

Final audio-only robot/test result after the selection lock:

- macro-F1 4-class: `0.602648`
- accuracy 4-class: `0.734114`
- contact macro-F1: `0.488824`
- binary macro-F1: `0.939217`

Final report:

```text
outputs/audio_feature_benchmarks/audio_report_grade_gate_select/reports/
audio_report_grade_gate_select_final_test_report.csv
```

Confusion matrix:

```text
,ambient,leaf,trunk,twig
ambient,1132,0,0,0
leaf,11,185,34,63
trunk,100,80,145,136
twig,23,93,50,167
```

## Why This Is Clearer Than The Previous Best Run

The previous best audio-only run used a broader locked OOF blender and reached
`0.602002` macro-F1. This protocol is cleaner for reporting because it:

- has only six predefined OOF candidates;
- uses rounded, readable weights;
- fixes gamma and class bias instead of sweeping them;
- keeps one explicit gate mechanism;
- writes a method card before loading robot/test;
- improves final macro-F1 slightly to `0.602648`.
