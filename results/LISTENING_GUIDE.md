# Listening Guide

This file highlights the main listening examples used in the notebook and the paper.

## Main Shared Comparison

These examples use the balanced inference profile, which is the main shared reference because both models remain monotonic under the same decoding setup.

* [Balanced continuous +1 semitone](./sweeps/balanced_sample_1_9s/continuous/continuous_up_1.wav)
* [Balanced categorical +1 semitone](./sweeps/balanced_sample_1_9s/categorical/categorical_up_1.wav)
* [Balanced continuous center](./sweeps/balanced_sample_1_9s/continuous/continuous_center.wav)
* [Balanced categorical center](./sweeps/balanced_sample_1_9s/categorical/categorical_center.wav)

## Baseline Reference

These files are useful when comparing the original long-run decoding setup against the balanced profile.

* [Baseline continuous +1 semitone](./sweeps/baseline_sample_3s/continuous/continuous_up_1.wav)
* [Baseline categorical +1 semitone](./sweeps/baseline_sample_3s/categorical/categorical_up_1.wav)

## Continuous Interpolation Showcase

These examples come from the cool profile. They are not the main shared comparison, but they best illustrate the interpolation behavior of the continuous model.

* [Cool continuous +0.25 semitone](./sweeps/cool_sample_1_9s/continuous/continuous_interp_up_0_25.wav)
* [Cool continuous +0.75 semitone](./sweeps/cool_sample_1_9s/continuous/continuous_interp_up_0_75.wav)

## Visual References

* [Pitch response curve](./figures/pitch_response_curve.png)
* [Onset metric comparison](./figures/onset_metric_comparison.png)
* [Onset STFT examples](./figures/onset_stft_examples.png)

## Quantitative Summaries

* [Inference sweep summary](./sweeps/summary.json)
* [Objective summary](./tables/objective_summary.json)
* [Pitch tracking report](./tables/pitch_tracking_report.json)
