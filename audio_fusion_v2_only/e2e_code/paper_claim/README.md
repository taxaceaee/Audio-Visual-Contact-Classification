# Paper claim path — pure single-shot architecture

This directory + scripts are the **only** path intended for a paper number.

## Why this exists

Exploratory runs under `outputs/audio_feature_benchmarks/multimodal_085_*` opened robot many times (multi-peek). Those are **non-claim**.

This path enforces:

1. **Frozen RECIPE** (`paper_claim/RECIPE.json`) with SHA-256 in the lock  
2. **Hand-only selection** → `selection_lock.json` (`test_loaded: false`)  
3. **One robot open** → `final_test_metrics.json` + irreversible `CLAIM_SEAL.json`  
4. **Second open refused** (unless `PAPER_CLAIM_FORCE_REOPEN=1` for non-claim diagnostics)

## Commands (exactly once)

```bash
# 1) Hand selection only (never loads robot)
python run_paper_claim_hand_selection.py

# 2) Single robot open + seal
python run_paper_claim_final_test_once.py
```

Outputs: `outputs/paper_claim_v1/`

## What is frozen vs selected

| Piece | Source |
|---|---|
| Contact primary/secondary thresholds | Frozen constants from **prior hand-only** locks (stress amb-lift + collapse-gain secondary). Not re-searched here. |
| Material bias `(T, b_leaf, b_trunk, b_twig)` | Selected **only on hand** by pre-registered cascade-gain ranking in RECIPE |
| Robot | Opened **once** after lock |

## Honesty boundary

Architectural single-shot ≠ erasing human multi-peek history on the same robot split.  
This path makes the **claim artifact** mechanically single-shot and prevents further peek-driven retuning on the claim directory.

For a stronger epistemic claim, re-run this same RECIPE on a **fresh holdout** never opened during exploration.

## Tests

```bash
pytest test_paper_claim_protocol.py -q
```
