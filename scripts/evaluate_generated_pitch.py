#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = ROOT / "results" / "audio_examples"
DEFAULT_OUTPUT_CSV = ROOT / "results" / "tables" / "pitch_tracking_summary.csv"
DEFAULT_OUTPUT_JSON = ROOT / "results" / "tables" / "pitch_tracking_report.json"

DISCRETE_ORDER = [
    "down_1",
    "down_0_5",
    "center",
    "up_0_5",
    "up_1",
]

INTERPOLATION_ORDER = [
    "down_1",
    "down_0_5",
    "center",
    "up_0_25",
    "up_0_5",
    "up_0_75",
    "up_1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua de forma simple el comportamiento de pitch en los audios generados."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio con generated_summary.json y subcarpetas de audio.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV de salida con estadisticas por archivo.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="JSON de salida con informe agregado.",
    )
    return parser.parse_args()


def load_summary(results_dir: Path) -> dict:
    summary_path = results_dir / "generated_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No existe {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def estimate_pitch_stats(audio_path: Path) -> dict:
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    f0 = librosa.yin(audio, fmin=55.0, fmax=1760.0, sr=sr)
    voiced = f0[np.isfinite(f0) & (f0 > 0)]

    if voiced.size == 0:
        return {
            "median_hz": None,
            "mean_hz": None,
            "std_hz": None,
            "voiced_frames": 0,
        }

    return {
        "median_hz": float(np.median(voiced)),
        "mean_hz": float(np.mean(voiced)),
        "std_hz": float(np.std(voiced)),
        "voiced_frames": int(voiced.size),
    }


def extract_label(file_name: str) -> str:
    stem = Path(file_name).stem
    if stem.startswith("continuous_interp_"):
        return stem.replace("continuous_interp_", "", 1)
    if stem.startswith("continuous_"):
        return stem.replace("continuous_", "", 1)
    if stem.startswith("categorical_"):
        return stem.replace("categorical_", "", 1)
    return stem


def monotonic_report(rows: list[dict]) -> dict:
    keyed = {}
    for row in rows:
        label = extract_label(row["file"])
        if label not in DISCRETE_ORDER:
            continue
        if row["median_hz"] is None:
            continue
        keyed[label] = row["median_hz"]

    ordered = [(label, keyed[label]) for label in DISCRETE_ORDER if label in keyed]
    is_monotonic = all(ordered[i][1] <= ordered[i + 1][1] for i in range(len(ordered) - 1))
    return {
        "ordered_median_hz": ordered,
        "is_monotonic": is_monotonic,
    }


def interpolation_report(rows: list[dict]) -> dict:
    keyed = {}
    for row in rows:
        label = extract_label(row["file"])
        if row["median_hz"] is None:
            continue
        keyed[label] = row["median_hz"]

    ordered = [(label, keyed[label]) for label in INTERPOLATION_ORDER if label in keyed]
    is_monotonic = all(ordered[i][1] <= ordered[i + 1][1] for i in range(len(ordered) - 1))

    between_0_25 = None
    between_0_75 = None

    if {"center", "up_0_25", "up_0_5"}.issubset(keyed):
        between_0_25 = keyed["center"] <= keyed["up_0_25"] <= keyed["up_0_5"]

    if {"up_0_5", "up_0_75", "up_1"}.issubset(keyed):
        between_0_75 = keyed["up_0_5"] <= keyed["up_0_75"] <= keyed["up_1"]

    return {
        "ordered_median_hz": ordered,
        "is_monotonic": is_monotonic,
        "up_0_25_between_neighbors": between_0_25,
        "up_0_75_between_neighbors": between_0_75,
    }


def main() -> None:
    args = parse_args()
    summary = load_summary(args.results_dir)

    rows = []
    report = {"models": {}}

    for model_name, items in summary.items():
        model_rows = []
        for item in items:
            wav_path = args.results_dir / model_name / item["file"]
            stats = estimate_pitch_stats(wav_path)
            row = {
                "model": model_name,
                "file": item["file"],
                "duration_sec": item["duration_sec"],
                "params": item["params"],
                **stats,
            }
            rows.append(row)
            model_rows.append(row)

        report["models"][model_name] = {
            "num_files": len(model_rows),
            "monotonic_discrete_sweep": monotonic_report(model_rows),
            "continuous_interpolation": interpolation_report(model_rows),
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model",
            "file",
            "duration_sec",
            "median_hz",
            "mean_hz",
            "std_hz",
            "voiced_frames",
            "params",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["params"] = json.dumps(serializable["params"], ensure_ascii=True)
            writer.writerow(serializable)

    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"CSV guardado en {args.output_csv}")
    print(f"JSON guardado en {args.output_json}")
    for model_name, info in report["models"].items():
        monotonic = info["monotonic_discrete_sweep"]["is_monotonic"]
        print(f"{model_name}: monotonic_discrete_sweep={monotonic}")
        interpolation = info["continuous_interpolation"]
        if interpolation["ordered_median_hz"]:
            print(
                f"{model_name}: interpolation_monotonic={interpolation['is_monotonic']} "
                f"up_0_25_between_neighbors={interpolation['up_0_25_between_neighbors']} "
                f"up_0_75_between_neighbors={interpolation['up_0_75_between_neighbors']}"
            )


if __name__ == "__main__":
    main()
