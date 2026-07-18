"""Evaluate sealed audio-only robot predictions (paper-clean lift blend).

Default path recomputes Macro F1 from locked robot predictions shipped in
results/. Does not retrain OOF audio sources (optional full retrain needs media).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow import of shared/ from release root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.metrics_utils import (  # noqa: E402
    assert_close,
    load_sealed_targets,
    metrics_from_predictions_csv,
)


def main() -> dict:
    section = Path(__file__).resolve().parents[1]
    pred_path = section / "results" / "predictions.csv"
    sealed_path = section / "results" / "metrics.json"
    targets = load_sealed_targets(_ROOT)["audio_only"]

    metrics = metrics_from_predictions_csv(pred_path)
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    sealed_m = sealed.get("metrics", sealed)

    assert_close(
        metrics["macro_f1_4class"],
        targets["macro_f1_4class"],
        name="audio macro_f1_4class vs target",
        tol=targets.get("tolerance", 1e-9),
    )
    assert_close(
        metrics["macro_f1_4class"],
        float(sealed_m["macro_f1_4class"]),
        name="audio macro_f1_4class vs shipped metrics.json",
        tol=1e-9,
    )
    assert metrics["n"] == targets["n"]

    out = {
        "section": "audio_only",
        "status": "ok",
        "source": str(pred_path.relative_to(_ROOT)),
        "metrics": metrics,
        "sealed_target_macro_f1_4class": targets["macro_f1_4class"],
    }
    print(json.dumps({
        "section": "audio_only",
        "macro_f1_4class": metrics["macro_f1_4class"],
        "accuracy_4class": metrics["accuracy_4class"],
        "binary_macro_f1": metrics["binary_macro_f1"],
        "contact_macro_f1": metrics["contact_macro_f1"],
        "match_sealed": True,
    }, indent=2))
    return out


if __name__ == "__main__":
    main()
