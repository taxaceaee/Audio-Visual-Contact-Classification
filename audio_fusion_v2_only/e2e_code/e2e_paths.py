"""Add bundle code roots to sys.path for E2E runs from any cwd."""
from __future__ import annotations

import sys
from pathlib import Path


def add_bundle_paths(bundle_root: Path | None = None) -> Path:
    here = Path(__file__).resolve().parent  # e2e_code
    root = bundle_root or here.parent
    roots = [
        root / "e2e_code",
        root / "audio_only" / "code" / "train_chain",
        root / "fusion_v2" / "code",
        root / "shared",
        root,
    ]
    for p in roots:
        s = str(p.resolve())
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)
    return root.resolve()


if __name__ == "__main__":
    print(add_bundle_paths())
