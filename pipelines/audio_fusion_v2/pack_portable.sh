#!/usr/bin/env bash
# Build a transfer archive of this bundle (self-contained, no symlinks).
# Run from repo root or from this folder:
#   bash audio_fusion_v2_only/pack_portable.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$(dirname "$HERE")}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="${OUT_DIR}/audio_fusion_v2_only_portable_${STAMP}.tar.gz"

# Refuse to pack if data still has symlinks
if find "$HERE/fusion_v2/data" -type l | grep -q .; then
  echo "ERROR: symlinks still present under fusion_v2/data — materialize first" >&2
  find "$HERE/fusion_v2/data" -type l >&2
  exit 1
fi

# Refresh SHA256SUMS for sealed artifacts + features
(
  cd "$HERE"
  {
    echo "# SHA256 of sealed eval artifacts + fusion features (portable integrity)"
    find audio_only/results fusion_v2/results shared/sealed_targets.json \
      fusion_v2/data/features fusion_v2/data/manifests \
      -type f \( -name '*.csv' -o -name '*.json' -o -name '*.npy' -o -name '*.npz' \) \
      | sort | while read -r f; do
        sha256sum "$f"
      done
  } > SHA256SUMS.txt
)

# Exclude caches / local run reports from archive (regenerated on target)
tar -C "$(dirname "$HERE")" \
  --exclude='audio_fusion_v2_only/**/__pycache__' \
  --exclude='audio_fusion_v2_only/**/*.pyc' \
  --exclude='audio_fusion_v2_only/.venv' \
  --exclude='audio_fusion_v2_only/reproduce_summary.json' \
  --exclude='audio_fusion_v2_only/verify_feature_order_report.json' \
  -czf "$ARCHIVE" "$(basename "$HERE")"

echo "Wrote $ARCHIVE"
ls -lh "$ARCHIVE"
echo "Extract elsewhere: tar -xzf $(basename "$ARCHIVE") && cd audio_fusion_v2_only && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 reproduce_two.py"
