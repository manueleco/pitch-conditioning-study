# Artifact Guide

This file summarizes the main entry points for the project repository.

## Main Narrative

* [official-notebook.ipynb](./official-notebook.ipynb) is the main walkthrough for the project.
* It shows dataset construction, representation checks, training summaries, objective analysis, sweep comparison, figures, and embedded audio playback.

## Core Scripts

* [scripts/build_two_bar_pitch_source.py](./scripts/build_two_bar_pitch_source.py) builds the short two-bar source dataset from the original Logic Pro recordings made in a home studio.
* [scripts/build_pitch_conditioning_variants.py](./scripts/build_pitch_conditioning_variants.py) derives the continuous and categorical conditioning variants from the same source audio.
* [scripts/run_dataprep_pipeline.py](./scripts/run_dataprep_pipeline.py) runs normalization, tokenization, sidecar generation, and Hugging Face dataset export.
* [scripts/train_pitch_models.py](./scripts/train_pitch_models.py) launches the paired training runs.
* [scripts/run_inference_sweep.py](./scripts/run_inference_sweep.py) executes the decoding sweep used in the paper.
* [scripts/evaluate_generated_pitch.py](./scripts/evaluate_generated_pitch.py) computes pitch-tracking summaries.
* [scripts/analyze_generated_audio.py](./scripts/analyze_generated_audio.py) computes objective audio metrics and saves the paper figures.

## Main Result Files

* [results/tables/objective_summary.json](./results/tables/objective_summary.json)
* [results/tables/pitch_tracking_report.json](./results/tables/pitch_tracking_report.json)
* [results/sweeps/summary.json](./results/sweeps/summary.json)
* [models/training_summary.json](./models/training_summary.json)

## Figures

* [results/figures/pitch_response_curve.png](./results/figures/pitch_response_curve.png)
* [results/figures/onset_metric_comparison.png](./results/figures/onset_metric_comparison.png)
* [results/figures/onset_stft_examples.png](./results/figures/onset_stft_examples.png)

## Listening Examples

* [results/LISTENING_GUIDE.md](./results/LISTENING_GUIDE.md) lists the main audio examples referenced in the notebook and the paper.
