# E2E upstream code — source-only bundle

The archive contains all source required by the v2 base path, but contains no
raw media, features, manifests, predictions, metrics, locks, or `outputs/`
caches. Supply or regenerate these external runtime inputs before execution.

## Required v2-base source chain

```text
train_val_select_final_test.py and audio train-chain sources
        -> run_segment_meta_weighted_group_selection.py
        -> run_segment_rule_stack_v2_group_selection.py
        -> run_segment_rule_stack_v2_group_final_test.py
        -> segment_rule_stack_v2_base_segment_outputs.npz (generated externally)
```

The first two selection stages load hand/default data only and write their
locks before the final-test runner can load robot/test data.

## Commands

```bash
export PYTHONPATH="e2e_code:audio_only/code/train_chain:fusion_v2/code:shared:${PYTHONPATH:-}"
python3 e2e_code/run_segment_meta_weighted_group_selection.py
python3 e2e_code/run_segment_rule_stack_v2_group_selection.py
python3 e2e_code/run_segment_rule_stack_v2_group_final_test.py
```

Extraction/training additionally needs the dataset root, raw media or valid
upstream features, and dependencies from `requirements.txt`. Build the archive
with `bash pack_code_only.sh`.
