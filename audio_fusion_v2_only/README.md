# audio_fusion_v2_only — portable sealed bundle

Self-contained folder for **audio-only** + **fusion paper claim v2** robot/test evaluation.

| Pipeline | Sealed Macro F1 (n=2219) |
|----------|-------------------------:|
| Audio-only | **0.7026719927789447** |
| Fusion v2 | **0.8507810400075331** |

No host absolute paths. No external symlinks. Features + manifests live **inside** this folder.

---

## Other machine (copy folder or unzip archive)

```bash
# 1) Copy this whole directory, or:
#    tar -xzf audio_fusion_v2_only_portable_*.tar.gz
cd audio_fusion_v2_only

# 2) Optional integrity check
sha256sum -c SHA256SUMS.txt   # if present

# 3) Isolated env + pinned deps
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# 4) Sealed reproduce (official entry)
python3 reproduce_two.py

# 5) Feature-order + fusion F1 checks (no parent workspace needed)
python3 verify_feature_order.py

# 6) Unit tests against shipped entry points
python3 -m pytest tests/ -q
```

Expected: exit code **0**, both Macro F1 values above within **1e-9**.

Optional env:

| Variable | Default | Meaning |
|----------|---------|---------|
| `TREE_BUNDLE_ROOT` | this folder | Override bundle root |
| `OMP_NUM_THREADS` etc. | set to `1` by scripts | Deterministic BLAS |
| `AFV2_CHECK_WORKSPACE_HASH=1` | off | Dev-only: hash features vs parent `outputs/` |

---

## Layout

```
audio_fusion_v2_only/
  reproduce_two.py              # official dual sealed entry
  verify_feature_order.py       # stack order + fusion F1
  pack_portable.sh              # build .tar.gz for transfer
  requirements.txt              # pinned numpy/pandas/sklearn
  SHA256SUMS.txt                # integrity (after pack / refresh)
  FILE_INVENTORY.md
  shared/                       # metrics_utils + sealed_targets
  audio_only/code/evaluate_audio.py
  audio_only/results/           # locked predictions
  audio_only/code/train_chain/  # reference only (needs full workspace)
  fusion_v2/code/paper_claim/clear_pipeline.py
  fusion_v2/code/evaluate_fusion.py
  fusion_v2/data/features/      # REAL files (total240, clip, w2v, v2_base)
  fusion_v2/data/manifests/     # REAL CSVs
  fusion_v2/results/            # CLAIM_SEAL + metrics
  tests/test_sealed_portable.py
```

## Feature stack (fusion)

```text
X_amb = hstack([ total240/robot_mix (240) , wav2vec2 (1536) , CLIP (768) ])  # dim 2544
```

Cascade: (1) v2 base → (2) amb-lift → (3) secondary CLIP trunk → (4) material `b_leaf=-1.3`.

## Pack for USB / other host

```bash
bash pack_portable.sh
# → ../audio_fusion_v2_only_portable_<UTC>.tar.gz
```

## Upstream / E2E assets (also in this zip)

| Path | Contents |
|------|----------|
| `outputs/audio_feature_benchmarks/` | OOF, highsr, pairwise, TTA, lift, v2 stack caches |
| `outputs/audio_wav2vec2_features/` | Pre-extracted wav2vec2 |
| `outputs/image_timm_features/...clip.../` | Pre-extracted CLIP |
| `e2e_code/` | v2 train, suite, claim scripts, total240/timm helpers |
| `E2E_UPSTREAM.md` | Detail on gaps vs full raw-media rebuild |

**Still external:** raw `.wav`/JPEG media (multi-GB). Sealed F1 does not need them.

## Not included (by design)

- Image-only pipeline as sealed section
- paper_claim v1 (0.853)
- Raw media corpus
