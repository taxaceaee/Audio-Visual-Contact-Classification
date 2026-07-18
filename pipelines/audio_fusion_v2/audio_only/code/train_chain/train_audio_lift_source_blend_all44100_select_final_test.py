from __future__ import annotations

from pathlib import Path

import train_audio_lift_source_blend_all16_select_final_test as impl


SAMPLE_RATE = 44100
RUN_SLUG = "audio_lift_source_blend_all44100_select"


def main() -> None:
    args = impl.parse_args()
    if args.output == impl.OUTPUT_ROOT:
        args.output = Path("outputs/audio_sample_rate_ablation/all44100")
    impl.SAMPLE_RATE = SAMPLE_RATE
    impl.RUN_SLUG = RUN_SLUG
    impl.HIGHSR_RUNS = {
        "highsr_default": ("audio_highsr_temporal_tta_select", "highsr_hgb_default__all_aug"),
        "highsr_regularized": ("audio_highsr_temporal_hgb_regularized_select", "highsr_hgb_regularized__all_aug"),
        "highsr_extratrees": ("audio_highsr_temporal_extratrees_select", "highsr_extratrees__all_aug"),
    }
    impl.run_experiment(args, SAMPLE_RATE, RUN_SLUG)


if __name__ == "__main__":
    main()
