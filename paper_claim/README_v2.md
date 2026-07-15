# Paper claim v2 — simplified single-shot

## Pipeline (no data leakage)

```
RECIPE_v2_simple.json (frozen, hashed)
        │
        ▼
run_paper_claim_v2_hand_selection.py   ← hand_train only, never robot
        │
        ▼
outputs/paper_claim_v2/selection_lock.json
        │
        ▼
run_paper_claim_v2_final_test_once.py  ← robot once + CLAIM_SEAL
```

## Architecture (simpler than v1)

| Piece | Detail |
|---|---|
| Base | v2 `hier_trunk_meta_else` |
| Amb-lift | `ts≥0.5`, `max(cs,bin)≥0.1` (stress multi-view detectors) |
| Secondary | CLIP trunk `th=0.625`, `cs=0.75` |
| **Dropped** | `do_tt`, `b_trunk`, `b_twig`, `T≠1`, `v2_rule_soft` |
| Material | protect-trunk; **only** `b_leaf` free; soft=`meta` |

## Hand selection (pre-registered)

- 1-D grid `b_leaf = linspace(-2, 0.5, 26)`
- Cascade surrogate (trunk→twig / twig→leaf) on hand labels only
- Eligible: `casc_gain≥0.015`, `clean≥0.94`, `nflip≥1`, trunk floor
- Rank: `casc_gain → clean → twig_rec`

## Sealed result (this run)

| Item | Value |
|---|---|
| Selected `b_leaf` | **−1.3** (hand) |
| Robot Macro F1 | **0.8508** (> 0.85) |
| Seal | `outputs/paper_claim_v2/CLAIM_SEAL.json` |
| Second open | **refused** |

## Leakage invariants

- Selection: no robot features/labels/manifest
- Groups: `specimen_group` (stem strip only)
- No class-from-filename features
- No robot UDA
- No HP tuning on robot F1
- Detectors fit on hand; applied to robot only after lock

## Commands

```bash
rm -rf outputs/paper_claim_v2   # only if starting a fresh claim version
python run_paper_claim_v2_hand_selection.py
python run_paper_claim_v2_final_test_once.py
pytest test_paper_claim_v2_no_leak.py -q
```

## Clear implementation (same F1, thinner code)

Detectors reduced to **CLIP ‖ wav2vec2 ‖ robot_mix** (drop multi-bb + bandlimit) — verified **identical** Macro F1 to seal:

```bash
python run_paper_claim_clear_verify.py
# clear_macro_f1 == sealed 0.8507810400075331
```

| File | Role |
|---|---|
| `paper_claim/clear_pipeline.py` | Single minimal pipeline |
| `paper_claim/ARCHITECTURE_CLEAR.md` | Mermaid + ASCII diagram |
| `paper_claim/architecture_clear.svg` | Slide-ready diagram |
| `run_paper_claim_clear_verify.py` | Match sealed claim |
