from __future__ import annotations

import sys

import train_audio_group_consistency_pair_blend_select_final_test as segment_impl


def main() -> None:
    # Log-probability consensus treats each window in a segment as an independent
    # piece of audio evidence and selects only the blend weight from train OOF.
    segment_impl.HIGHSR_WEIGHT_CANDIDATES[:] = [0.65, 0.70, 0.75, 0.80]
    segment_impl.GROUP_RULES[:] = ["sum_log_proba"]

    if "--run-slug" not in sys.argv:
        sys.argv.extend(["--run-slug", "audio_log_consensus_pair_blend_select"])

    segment_impl.main()


if __name__ == "__main__":
    main()
