#!/usr/bin/env bash
# Build the source-only archive for the v2 fusion pipeline.
# No feature, manifest, prediction, metric, or other data artifact is copied.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$(dirname "$HERE")/audio_fusion_v2_only_code.zip}"
STAGE="$(mktemp -d)"
ROOT="$STAGE/audio_fusion_v2_only"
trap 'rm -rf "$STAGE"' EXIT

required=(
  'e2e_code/run_segment_meta_weighted_group_selection.py'
  'e2e_code/run_segment_rule_stack_v2_group_selection.py'
  'e2e_code/run_segment_rule_stack_v2_group_final_test.py'
  'audio_only/code/train_chain/train_audio_broad_oof_meta_select_final_test.py'
  'audio_only/code/train_chain/train_audio_group_consistency_pair_blend_select_final_test.py'
  'audio_only/code/train_chain/train_audio_lift_source_blend_select_final_test.py'
  'audio_only/code/train_chain/train_audio_specimen_contact_consensus_select_final_test.py'
  'audio_only/code/train_chain/train_val_select_final_test.py'
)
for rel in "${required[@]}"; do
  [[ -f "$HERE/$rel" ]] || { echo "ERROR: missing source $rel" >&2; exit 1; }
done

mkdir -p "$ROOT/audio_only" "$ROOT/fusion_v2" "$ROOT/shared" "$ROOT/tests"
cp -a "$HERE/e2e_code" "$ROOT/"
cp -a "$HERE/audio_only/code" "$ROOT/audio_only/"
cp -a "$HERE/fusion_v2/code" "$ROOT/fusion_v2/"
cp -a "$HERE/shared/metrics_utils.py" "$ROOT/shared/"
cp -a "$HERE/tests/test_sealed_portable.py" "$ROOT/tests/"
cp -a "$HERE/reproduce_two.py" "$HERE/verify_feature_order.py" "$ROOT/"
cp -a "$HERE/requirements.txt" "$HERE/README.md" "$HERE/E2E_UPSTREAM.md" "$HERE/FILE_INVENTORY.md" "$ROOT/"
cp -a "$HERE/pack_code_only.sh" "$ROOT/"

if find "$ROOT" -type f \( -name '*.npy' -o -name '*.npz' -o -name '*.csv' \) -print -quit | grep -q .; then
  echo 'ERROR: data leaked into code-only archive' >&2
  exit 1
fi

# zip updates existing archives by default; remove the explicit target first so
# stale feature entries from an older bundle cannot survive this build.
rm -f "$OUT"

(
  cd "$STAGE"
  find audio_fusion_v2_only -type f -name '*.py' -print0 | xargs -0 -r python3 -m py_compile
  find audio_fusion_v2_only -type d -name '__pycache__' -prune -exec rm -rf {} +
  find audio_fusion_v2_only -type f | sed 's#^#/#' | sort > audio_fusion_v2_only/CODE_ONLY_MANIFEST.txt
  zip -qr "$OUT" audio_fusion_v2_only
)

unzip -tqq "$OUT"
if unzip -Z1 "$OUT" | rg -i '\.(npy|npz|csv)$'; then
  echo 'ERROR: feature/data artifact present in archive' >&2
  exit 1
fi
echo "Wrote $OUT"
echo "Source files: $(unzip -Z1 "$OUT" | wc -l)"
