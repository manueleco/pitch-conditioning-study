#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_DIR = ROOT / ".tmp" / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import librosa
import librosa.display
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = ROOT / "results" / "audio_examples"
DEFAULT_REFERENCE_DIR = ROOT / "data" / "two_bar_continuous_pitch" / "raw" / "train"
DEFAULT_OUTPUT_CSV = ROOT / "results" / "tables" / "objective_metrics.csv"
DEFAULT_OUTPUT_JSON = ROOT / "results" / "tables" / "objective_summary.json"
DEFAULT_FIGURES_DIR = ROOT / "results" / "figures"

TARGET_BY_LABEL = {
    "down_1": -1.0,
    "down_0_5": -0.5,
    "center": 0.0,
    "up_0_25": 0.25,
    "up_0_5": 0.5,
    "up_0_75": 0.75,
    "up_1": 1.0,
}

DISCRETE_LABELS = ["down_1", "down_0_5", "center", "up_0_5", "up_1"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcula metricas objetivas sobre los audios generados y guarda figuras de apoyo."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def load_summary(results_dir: Path):
    summary_path = results_dir / "generated_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No existe {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_audio(audio_path: Path):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    return audio.astype(np.float32), sr


def extract_label(file_name: str):
    stem = Path(file_name).stem
    prefixes = [
        "continuous_interp_",
        "continuous_",
        "categorical_",
    ]
    for prefix in prefixes:
        if stem.startswith(prefix):
            return stem.replace(prefix, "", 1)
    return stem


def target_shift_for_item(item: dict):
    params = item.get("params", {})
    if "pitch_shift" in params:
        return float(params["pitch_shift"])
    label = extract_label(item["file"])
    return TARGET_BY_LABEL.get(label)


def voiced_pitch_stats(audio: np.ndarray, sr: int):
    f0 = librosa.yin(audio, fmin=55.0, fmax=1760.0, sr=sr)
    voiced = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced.size == 0:
        return {
            "median_hz": None,
            "mean_hz": None,
            "std_hz": None,
            "pitch_std_semitones": None,
            "voiced_frames": 0,
        }

    median_hz = float(np.median(voiced))
    semitone_offsets = 12.0 * np.log2(voiced / median_hz)
    return {
        "median_hz": median_hz,
        "mean_hz": float(np.mean(voiced)),
        "std_hz": float(np.std(voiced)),
        "pitch_std_semitones": float(np.std(semitone_offsets)),
        "voiced_frames": int(voiced.size),
    }


def high_band_share(audio: np.ndarray, sr: int, cutoff_hz: float = 4000.0):
    if audio.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sr)
    total = float(np.sum(spectrum)) + 1e-8
    high = float(np.sum(spectrum[freqs >= cutoff_hz]))
    return high / total


def safe_segment(audio: np.ndarray, start: int, end: int):
    start = max(0, min(start, audio.size))
    end = max(start + 1, min(end, audio.size))
    return audio[start:end]


def onset_metrics(audio: np.ndarray, sr: int):
    onset = safe_segment(audio, 0, int(0.05 * sr))
    body = safe_segment(audio, int(0.10 * sr), int(0.35 * sr))
    onset_n_fft = max(64, min(512, onset.size))
    body_n_fft = max(64, min(512, body.size))

    onset_rms = float(np.sqrt(np.mean(np.square(onset))) + 1e-8)
    body_rms = float(np.sqrt(np.mean(np.square(body))) + 1e-8)

    onset_centroid = float(np.mean(librosa.feature.spectral_centroid(y=onset, sr=sr, n_fft=onset_n_fft, hop_length=max(1, onset_n_fft // 4))))
    body_centroid = float(np.mean(librosa.feature.spectral_centroid(y=body, sr=sr, n_fft=body_n_fft, hop_length=max(1, body_n_fft // 4))))

    onset_flatness = float(np.mean(librosa.feature.spectral_flatness(y=onset, n_fft=onset_n_fft, hop_length=max(1, onset_n_fft // 4))))
    body_flatness = float(np.mean(librosa.feature.spectral_flatness(y=body, n_fft=body_n_fft, hop_length=max(1, body_n_fft // 4))))

    onset_high = high_band_share(onset, sr=sr)
    body_high = high_band_share(body, sr=sr)

    return {
        "onset_rms": onset_rms,
        "body_rms": body_rms,
        "onset_rms_ratio": onset_rms / body_rms,
        "onset_centroid_hz": onset_centroid,
        "body_centroid_hz": body_centroid,
        "onset_centroid_ratio": onset_centroid / (body_centroid + 1e-8),
        "onset_flatness": onset_flatness,
        "body_flatness": body_flatness,
        "onset_flatness_ratio": onset_flatness / (body_flatness + 1e-8),
        "onset_high_band_share": onset_high,
        "body_high_band_share": body_high,
        "onset_high_band_ratio": onset_high / (body_high + 1e-8),
    }


def analyze_file(audio_path: Path):
    audio, sr = load_audio(audio_path)
    metrics = {
        "file": audio_path.name,
        "sr": sr,
        "duration_sec": float(audio.size / sr),
    }
    metrics.update(voiced_pitch_stats(audio, sr))
    metrics.update(onset_metrics(audio, sr))
    return metrics


def mean_or_none(values: list[float | None]):
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def build_reference_summary(reference_dir: Path):
    reference_files = sorted(reference_dir.glob("*__pitch_center.wav"))
    rows = [analyze_file(path) for path in reference_files]
    summary = {
        "num_files": len(rows),
        "onset_rms_ratio_mean": mean_or_none([row["onset_rms_ratio"] for row in rows]),
        "onset_centroid_ratio_mean": mean_or_none([row["onset_centroid_ratio"] for row in rows]),
        "onset_flatness_ratio_mean": mean_or_none([row["onset_flatness_ratio"] for row in rows]),
        "onset_high_band_ratio_mean": mean_or_none([row["onset_high_band_ratio"] for row in rows]),
    }
    return {"rows": rows, "summary": summary}


def relative_shift_semitones(median_hz: float, center_hz: float):
    return float(12.0 * np.log2(median_hz / center_hz))


def create_pitch_response_figure(rows_by_model: dict[str, list[dict]], figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for model_name, rows in rows_by_model.items():
        ordered = sorted(
            [
                row for row in rows
                if row.get("target_shift") is not None and row.get("estimated_shift") is not None
            ],
            key=lambda row: row["target_shift"],
        )
        x = [row["target_shift"] for row in ordered]
        y = [row["estimated_shift"] for row in ordered]
        plt.plot(x, y, marker="o", label=model_name)

    ref = np.array(sorted(TARGET_BY_LABEL.values()), dtype=float)
    plt.plot(ref, ref, linestyle="--", color="black", alpha=0.6, label="ideal")
    plt.xlabel("Requested pitch shift (semitones)")
    plt.ylabel("Estimated pitch shift relative to center (semitones)")
    plt.title("Pitch response")
    plt.grid(alpha=0.25)
    plt.legend()
    out_path = figures_dir / "pitch_response_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def create_onset_metric_figure(model_summary: dict, reference_summary: dict, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "onset_rms_ratio_mean",
        "onset_centroid_ratio_mean",
        "onset_flatness_ratio_mean",
        "onset_high_band_ratio_mean",
    ]
    labels = ["RMS", "Centroid", "Flatness", "High band"]

    reference_values = [reference_summary.get(name) or 0.0 for name in metric_names]
    continuous_values = [model_summary["continuous"].get(name) or 0.0 for name in metric_names]
    categorical_values = [model_summary["categorical"].get(name) or 0.0 for name in metric_names]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width, reference_values, width=width, label="real_center_train")
    plt.bar(x, continuous_values, width=width, label="continuous")
    plt.bar(x + width, categorical_values, width=width, label="categorical")
    plt.xticks(x, labels)
    plt.ylabel("Onset to body ratio")
    plt.title("Onset metric comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    out_path = figures_dir / "onset_metric_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def create_onset_spectrograms(results_dir: Path, reference_dir: Path, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        ("continuous", results_dir / "continuous" / "continuous_center.wav"),
        ("categorical", results_dir / "categorical" / "categorical_center.wav"),
    ]
    real_candidates = sorted(reference_dir.glob("*__pitch_center.wav"))
    if real_candidates:
        candidates.append(("real", real_candidates[0]))

    fig, axes = plt.subplots(len(candidates), 2, figsize=(10, 3.2 * len(candidates)))
    if len(candidates) == 1:
        axes = np.array([axes])

    for row_idx, (label, audio_path) in enumerate(candidates):
        audio, sr = load_audio(audio_path)
        zoom = safe_segment(audio, 0, int(0.15 * sr))
        time_axis = np.arange(zoom.size) / sr
        axes[row_idx, 0].plot(time_axis, zoom, linewidth=0.8)
        axes[row_idx, 0].set_title(f"{label} waveform 0-150 ms")
        axes[row_idx, 0].set_xlabel("s")
        axes[row_idx, 0].set_ylabel("amp")

        n_fft = max(64, min(512, zoom.size))
        hop_length = max(1, n_fft // 8)
        stft = np.abs(librosa.stft(zoom, n_fft=n_fft, hop_length=hop_length))
        db = librosa.amplitude_to_db(stft + 1e-8, ref=np.max)
        img = librosa.display.specshow(
            db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="linear",
            ax=axes[row_idx, 1],
        )
        axes[row_idx, 1].set_title(f"{label} STFT 0-150 ms")
        fig.colorbar(img, ax=axes[row_idx, 1], format="%+2.0f dB")

    out_path = figures_dir / "onset_stft_examples.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def main():
    args = parse_args()
    summary = load_summary(args.results_dir)
    reference = build_reference_summary(args.reference_dir)

    rows = []
    rows_by_model: dict[str, list[dict]] = {}

    for model_name, items in summary.items():
        model_rows = []
        for item in items:
            audio_path = args.results_dir / model_name / item["file"]
            row = analyze_file(audio_path)
            row["model"] = model_name
            row["label"] = extract_label(item["file"])
            row["target_shift"] = target_shift_for_item(item)
            row["params"] = item.get("params", {})
            model_rows.append(row)
        rows_by_model[model_name] = model_rows

    for model_name, model_rows in rows_by_model.items():
        center_row = next((row for row in model_rows if row["label"] == "center"), None)
        center_hz = center_row["median_hz"] if center_row else None
        for row in model_rows:
            if center_hz is not None and row["median_hz"] is not None:
                row["estimated_shift"] = relative_shift_semitones(row["median_hz"], center_hz)
            else:
                row["estimated_shift"] = None
            if row["target_shift"] is not None and row["estimated_shift"] is not None:
                row["shift_error"] = row["estimated_shift"] - row["target_shift"]
                row["abs_shift_error"] = abs(row["shift_error"])
            else:
                row["shift_error"] = None
                row["abs_shift_error"] = None
            rows.append(row)

    model_summary = {}
    for model_name, model_rows in rows_by_model.items():
        discrete_rows = [row for row in model_rows if row["label"] in DISCRETE_LABELS]
        all_shift_rows = [row for row in model_rows if row["abs_shift_error"] is not None]

        model_summary[model_name] = {
            "num_files": len(model_rows),
            "pitch_mae_discrete_semitones": mean_or_none([row["abs_shift_error"] for row in discrete_rows]),
            "pitch_mae_all_semitones": mean_or_none([row["abs_shift_error"] for row in all_shift_rows]),
            "pitch_stability_mean_semitones": mean_or_none([row["pitch_std_semitones"] for row in model_rows]),
            "onset_rms_ratio_mean": mean_or_none([row["onset_rms_ratio"] for row in model_rows]),
            "onset_centroid_ratio_mean": mean_or_none([row["onset_centroid_ratio"] for row in model_rows]),
            "onset_flatness_ratio_mean": mean_or_none([row["onset_flatness_ratio"] for row in model_rows]),
            "onset_high_band_ratio_mean": mean_or_none([row["onset_high_band_ratio"] for row in model_rows]),
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "file",
        "label",
        "duration_sec",
        "median_hz",
        "mean_hz",
        "std_hz",
        "pitch_std_semitones",
        "voiced_frames",
        "target_shift",
        "estimated_shift",
        "shift_error",
        "abs_shift_error",
        "onset_rms_ratio",
        "onset_centroid_ratio",
        "onset_flatness_ratio",
        "onset_high_band_ratio",
        "params",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {key: row.get(key) for key in fieldnames}
            serializable["params"] = json.dumps(row.get("params", {}), ensure_ascii=True)
            writer.writerow(serializable)

    figure_paths = {
        "pitch_response_curve": str(create_pitch_response_figure(rows_by_model, args.figures_dir)),
        "onset_metric_comparison": str(create_onset_metric_figure(model_summary, reference["summary"], args.figures_dir)),
        "onset_stft_examples": str(create_onset_spectrograms(args.results_dir, args.reference_dir, args.figures_dir)),
    }

    payload = {
        "reference_center_train": reference["summary"],
        "models": model_summary,
        "figures": figure_paths,
    }
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"CSV guardado en {args.output_csv}")
    print(f"JSON guardado en {args.output_json}")
    for key, value in figure_paths.items():
        print(f"Figura guardada {key}: {value}")
    for model_name, info in model_summary.items():
        print(
            f"{model_name}: pitch_mae_discrete={info['pitch_mae_discrete_semitones']:.4f} "
            f"pitch_mae_all={info['pitch_mae_all_semitones']:.4f} "
            f"onset_flatness_ratio_mean={info['onset_flatness_ratio_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
