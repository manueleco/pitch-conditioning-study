#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pitch_conditioning_study.pitch_levels import load_pitch_levels


VALID_SPLITS = ("train", "validation", "test")
RE_INDEX = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye un dataset corto de dos compases a partir de las tomas originales."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "two_bar_dataset.json",
        help="Archivo JSON con la configuracion del dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reescribe la carpeta de salida si ya existe.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent.parent if config_path.parent.name == "configs" else ROOT

    def resolve(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (base / path).resolve()

    data["source_audio_dir"] = resolve(data["source_audio_dir"])
    data["output_root"] = resolve(data["output_root"])
    data["pitch_levels_config"] = resolve(data["pitch_levels_config"])
    return data


def ensure_output(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def natural_key(path: Path) -> tuple[int, str]:
    match = RE_INDEX.search(path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def detect_start_sec(audio: np.ndarray, sr: int, top_db: float) -> float:
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return 0.0
    return intervals[0][0] / sr


def crop_clip(audio: np.ndarray, sr: int, start_sec: float, duration_sec: float) -> np.ndarray:
    start = max(0, int(round(start_sec * sr)))
    target = int(round(duration_sec * sr))
    end = min(len(audio), start + target)
    clip = audio[start:end]

    if len(clip) < target:
        pad = np.zeros(target - len(clip), dtype=np.float32)
        clip = np.concatenate([clip.astype(np.float32, copy=False), pad], axis=0)

    return clip.astype(np.float32, copy=False)


def write_annotation(csv_path: Path, pitch_value: float, fps: int, duration_sec: float) -> int:
    frames = max(1, int(round(duration_sec * fps)))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pitch_shift"])
        for _ in range(frames):
            writer.writerow([f"{pitch_value:.1f}"])
    return frames


def write_audio(audio_path: Path, audio: np.ndarray, sr: int) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, audio, sr, subtype="PCM_16")


def split_source_files(files: list[Path], split_counts: dict[str, int], showcase_count: int, seed: int) -> tuple[dict[str, list[Path]], list[Path]]:
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)

    required = sum(split_counts.values()) + showcase_count
    if len(shuffled) < required:
        raise ValueError(
            f"No hay suficientes tomas base. Requeridas={required}, disponibles={len(shuffled)}"
        )

    splits = {}
    cursor = 0
    for split in VALID_SPLITS:
        count = int(split_counts[split])
        splits[split] = sorted(shuffled[cursor:cursor + count], key=natural_key)
        cursor += count

    showcase = sorted(shuffled[cursor:cursor + showcase_count], key=natural_key)
    return splits, showcase


def write_parameters(output_root: Path, levels) -> None:
    params = {
        "parameter_1": {
            "name": "pitch_shift",
            "type": "continuous",
            "unit": "semitones",
            "min": min(level.value for level in levels),
            "max": max(level.value for level in levels),
        }
    }
    (output_root / "parameters.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def export_group(
    output_dir: Path,
    files: list[Path],
    levels,
    clip_duration_sec: float,
    pre_roll_sec: float,
    top_db: float,
    fps: int,
) -> tuple[list[dict], Counter]:
    metadata = []
    pitch_counter = Counter()

    for wav_path in files:
        audio, sr = librosa.load(wav_path, sr=None, mono=True)
        start_sec = max(0.0, detect_start_sec(audio, sr, top_db=top_db) - pre_roll_sec)
        base_clip = crop_clip(audio, sr, start_sec=start_sec, duration_sec=clip_duration_sec)

        for level in levels:
            if abs(level.value) < 1e-9:
                shifted = base_clip
            else:
                shifted = librosa.effects.pitch_shift(base_clip, sr=sr, n_steps=level.value)
                shifted = shifted.astype(np.float32, copy=False)

            stem = f"{wav_path.stem}__pitch_{level.label}"
            audio_path = output_dir / f"{stem}.wav"
            csv_path = output_dir / f"{stem}.csv"

            write_audio(audio_path, shifted, sr)
            frame_count = write_annotation(csv_path, pitch_value=level.value, fps=fps, duration_sec=clip_duration_sec)

            metadata.append(
                {
                    "source_file": wav_path.name,
                    "output_file": audio_path.name,
                    "pitch_label": level.label,
                    "pitch_value": level.value,
                    "crop_start_sec": round(start_sec, 6),
                    "clip_duration_sec": clip_duration_sec,
                    "sample_rate": sr,
                    "annotation_frames": frame_count,
                }
            )
            pitch_counter[level.label] += 1

    return metadata, pitch_counter


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_root = cfg["output_root"]

    ensure_output(output_root, overwrite=args.overwrite)
    levels = load_pitch_levels(cfg["pitch_levels_config"])
    write_parameters(output_root, levels)

    excluded = set(cfg.get("excluded_files", []))
    source_files = [
        path
        for path in sorted(cfg["source_audio_dir"].glob("*.wav"), key=natural_key)
        if path.name not in excluded
    ]
    splits, showcase_files = split_source_files(
        source_files,
        split_counts=cfg["split_counts"],
        showcase_count=int(cfg["showcase_count"]),
        seed=int(cfg["seed"]),
    )

    summary = {
        "config": {
            "clip_duration_sec": cfg["clip_duration_sec"],
            "pre_roll_sec": cfg["pre_roll_sec"],
            "split_top_db": cfg["split_top_db"],
            "fps": cfg["fps"],
            "seed": cfg["seed"],
            "excluded_files": sorted(excluded),
        },
        "levels": [
            {"label": level.label, "value": level.value, "display": level.display}
            for level in levels
        ],
        "splits": {},
        "showcase": {},
    }

    raw_root = output_root / "raw"
    for split, files in splits.items():
        out_dir = raw_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata, pitch_counter = export_group(
            output_dir=out_dir,
            files=files,
            levels=levels,
            clip_duration_sec=float(cfg["clip_duration_sec"]),
            pre_roll_sec=float(cfg["pre_roll_sec"]),
            top_db=float(cfg["split_top_db"]),
            fps=int(cfg["fps"]),
        )
        summary["splits"][split] = {
            "source_files": [p.name for p in files],
            "num_source_files": len(files),
            "num_generated_files": len(metadata),
            "pitch_counts": dict(pitch_counter),
            "metadata_file": f"metadata_{split}.json",
        }
        (output_root / f"metadata_{split}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"Split {split}: fuentes={len(files)} generados={len(metadata)} niveles={dict(pitch_counter)}")

    showcase_dir = output_root / "showcase"
    showcase_dir.mkdir(parents=True, exist_ok=True)
    showcase_metadata, showcase_counter = export_group(
        output_dir=showcase_dir,
        files=showcase_files,
        levels=levels,
        clip_duration_sec=float(cfg["clip_duration_sec"]),
        pre_roll_sec=float(cfg["pre_roll_sec"]),
        top_db=float(cfg["split_top_db"]),
        fps=int(cfg["fps"]),
    )
    summary["showcase"] = {
        "source_files": [p.name for p in showcase_files],
        "num_source_files": len(showcase_files),
        "num_generated_files": len(showcase_metadata),
        "pitch_counts": dict(showcase_counter),
        "metadata_file": "metadata_showcase.json",
    }
    (output_root / "metadata_showcase.json").write_text(
        json.dumps(showcase_metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Showcase: fuentes={len(showcase_files)} generados={len(showcase_metadata)} niveles={dict(showcase_counter)}")

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Dataset base creado en {output_root}")


if __name__ == "__main__":
    main()
