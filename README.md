# Audio-Visual Contact Classification

Audio–visual contact material classification for tree structures in agriculture  
(4-class: **ambient / leaf / trunk / twig**).

| Pipeline | Robot Macro F1 (n=2219) | Notes |
|----------|------------------------:|-------|
| **Fusion paper claim v2** | **0.850781** | Sealed pure single-shot cascade |
| Audio-only | **0.702672** | Locked lift / source blend |
| Image-only (reference) | ~0.45 | Not required for fusion claim |

---

## Repository layout

```text
.
├── pipelines/audio_fusion_v2/   # ★ sealed package (code + results; no feature blobs)
│   ├── audio_only/              # evaluate + train_chain reference + sealed results
│   ├── fusion_v2/               # clear_pipeline, evaluate, CLAIM_SEAL metrics
│   ├── e2e_code/                # claim runners + segment stack v2 upstream
│   ├── shared/                  # metrics_utils, sealed_targets.json
│   ├── tests/
│   ├── reproduce_two.py         # official dual sealed entry
│   └── README.md
├── paper_claim/                 # importable clear claim package (workspace copy)
├── experiments/                 # research scripts (not the paper claim path)
│   ├── audio/
│   ├── image/
│   ├── multimodal/
│   ├── segment/
│   ├── paper_claim/
│   └── misc/
├── docs/
│   ├── paper/                   # PAPER_CLAIM_V2_FUSION_GUIDE
│   ├── guides/
│   └── attempts/                # attempt logs & protocol catalogs
├── notebooks/
├── tools/                       # diagrams, report helpers
└── requirements.txt             # see pipelines/audio_fusion_v2/requirements.txt
```

**Paper / report path:** use `pipelines/audio_fusion_v2/` only.  
**Do not** commit `.zip` archives or precomputed `.npy` features (see `.gitignore`).

---

## Quick start (sealed reproduce)

Features for full fusion re-run are **not** in this git tree (too large).  
This repo ships **code + sealed metrics**. With a portable feature bundle:

```bash
cd pipelines/audio_fusion_v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 reproduce_two.py   # expects bundled features if present
```

Sealed targets (also under `pipelines/audio_fusion_v2/shared/sealed_targets.json`):

- Fusion v2 Macro F1: `0.8507810400075331`
- Audio-only Macro F1: `0.7026719927789447`

Results on git:

- `pipelines/audio_fusion_v2/fusion_v2/results/` — `CLAIM_SEAL.json`, `final_test_metrics.json`
- `pipelines/audio_fusion_v2/audio_only/results/` — metrics, predictions, confusion

---

## Paper claim v2 (architecture)

Cascade (clear):

1. Audio segment v2 base (`hier_trunk_meta_else`)
2. Amb-lift → trunk (stress detectors: total240 + wav2vec2 + CLIP)
3. Secondary image trunk lift
4. Material redecode with locked `b_leaf = −1.3` (protect trunk)

Details: [`docs/paper/PAPER_CLAIM_V2_FUSION_GUIDE.md`](docs/paper/PAPER_CLAIM_V2_FUSION_GUIDE.md)  
Clear code: `pipelines/audio_fusion_v2/fusion_v2/code/paper_claim/clear_pipeline.py`

---

## Branch note

Primary packaging work lives on `feature/audio-fusion-v2-only`.  
`experiments/` holds historical ablations; they are **not** the sealed paper number.
