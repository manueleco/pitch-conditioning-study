#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pitch_conditioning_study.pitch_levels import load_pitch_levels, quantize_pitch


DEFAULT_SOURCE_ROOT = ROOT / "data" / "two_bar_pitch_source"
DEFAULT_CONTINUOUS_ROOT = ROOT / "data" / "two_bar_continuous_pitch"
DEFAULT_CATEGORICAL_ROOT = ROOT / "data" / "two_bar_categorical_pitch"
DEFAULT_LEVELS_PATH = ROOT / "configs" / "pitch_levels.json"
VALID_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye variantes continua y categorica del dataset de pitch."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Ruta del dataset base en formato raw con splits.",
    )
    parser.add_argument(
        "--continuous-root",
        type=Path,
        default=DEFAULT_CONTINUOUS_ROOT,
        help="Ruta de salida para la variante continua.",
    )
    parser.add_argument(
        "--categorical-root",
        type=Path,
        default=DEFAULT_CATEGORICAL_ROOT,
        help="Ruta de salida para la variante categorica.",
    )
    parser.add_argument(
        "--levels-config",
        type=Path,
        default=DEFAULT_LEVELS_PATH,
        help="Archivo JSON con los niveles discretos de pitch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reescribe las carpetas de salida si ya existen.",
    )
    return parser.parse_args()


def ensure_clean_output(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_source_parameters(source_root: Path) -> dict:
    parameters_path = source_root / "parameters.json"
    if not parameters_path.exists():
        raise FileNotFoundError(f"No se encontro parameters.json en {parameters_path}")
    return json.loads(parameters_path.read_text(encoding="utf-8"))


def resolve_split_root(source_root: Path) -> Path:
    raw_root = source_root / "raw"
    if raw_root.exists():
        return raw_root
    return source_root


def build_variant_parameters(source_parameters: dict, mode: str, levels_config: dict) -> dict:
    values = [float(item["value"]) for item in levels_config["levels"]]
    labels = [str(item["label"]) for item in levels_config["levels"]]

    output = {}
    for key, param_info in source_parameters.items():
        copied = dict(param_info)
        if copied.get("name") == "pitch_shift":
            if mode == "continuous":
                copied["type"] = "continuous"
                copied["min"] = min(values)
                copied["max"] = max(values)
                copied.pop("classes", None)
                copied.pop("num_classes", None)
            elif mode == "categorical":
                copied["type"] = "class"
                copied["classes"] = labels
                copied.pop("min", None)
                copied.pop("max", None)
                copied.pop("num_classes", None)
            else:
                raise ValueError(f"Modo no soportado: {mode}")
        output[key] = copied
    return output


def rewrite_csv(input_csv: Path, output_csv: Path, levels, mode: str) -> Counter:
    counts = Counter()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8", newline="") as handle_in:
        reader = csv.DictReader(handle_in)
        fieldnames = reader.fieldnames or []
        if "pitch_shift" not in fieldnames:
            raise KeyError(f"El archivo {input_csv} no contiene la columna pitch_shift")

        rows = []
        for row in reader:
            quantized = quantize_pitch(float(row["pitch_shift"]), levels)
            counts[quantized.label] += 1
            row["pitch_shift"] = (
                f"{quantized.value:.1f}" if mode == "continuous" else quantized.label
            )
            rows.append(row)

    with output_csv.open("w", encoding="utf-8", newline="") as handle_out:
        writer = csv.DictWriter(handle_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return counts


def copy_split(source_root: Path, output_root: Path, levels, mode: str) -> dict:
    raw_root = output_root / "raw"
    source_split_root = resolve_split_root(source_root)
    split_summary = {}

    for split in VALID_SPLITS:
        split_dir = source_split_root / split
        if not split_dir.exists():
            continue

        out_split_dir = raw_root / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        split_counter = Counter()
        file_counter = 0

        for wav_path in sorted(split_dir.glob("*.wav")):
            csv_path = wav_path.with_suffix(".csv")
            if not csv_path.exists():
                continue

            shutil.copy2(wav_path, out_split_dir / wav_path.name)
            counts = rewrite_csv(csv_path, out_split_dir / csv_path.name, levels, mode)
            split_counter.update(counts)
            file_counter += 1

        split_summary[split] = {
            "files": file_counter,
            "pitch_counts": dict(split_counter),
        }

    return split_summary


def save_metadata(output_root: Path, parameters: dict, levels_config: dict, split_summary: dict, mode: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "mode": mode,
        "pitch_levels": levels_config["levels"],
        "splits": split_summary,
    }
    (output_root / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_variant(
    source_root: Path,
    output_root: Path,
    levels,
    levels_config: dict,
    source_parameters: dict,
    mode: str,
    overwrite: bool,
) -> None:
    ensure_clean_output(output_root, overwrite=overwrite)
    split_summary = copy_split(source_root, output_root, levels, mode=mode)
    parameters = build_variant_parameters(source_parameters, mode=mode, levels_config=levels_config)
    save_metadata(output_root, parameters, levels_config, split_summary, mode=mode)

    print(f"Variante preparada: {output_root}")
    for split, info in split_summary.items():
        print(f"  split={split} archivos={info['files']} niveles={info['pitch_counts']}")


def main() -> None:
    args = parse_args()

    if not args.source_root.exists():
        raise FileNotFoundError(f"No existe el dataset base: {args.source_root}")

    levels_config = json.loads(args.levels_config.read_text(encoding="utf-8"))
    levels = load_pitch_levels(args.levels_config)
    source_parameters = load_source_parameters(args.source_root)

    print("Construyendo variante continua")
    build_variant(
        source_root=args.source_root,
        output_root=args.continuous_root,
        levels=levels,
        levels_config=levels_config,
        source_parameters=source_parameters,
        mode="continuous",
        overwrite=args.overwrite,
    )

    print("Construyendo variante categorica")
    build_variant(
        source_root=args.source_root,
        output_root=args.categorical_root,
        levels=levels,
        levels_config=levels_config,
        source_parameters=source_parameters,
        mode="categorical",
        overwrite=args.overwrite,
    )

    print("Proceso completado")


if __name__ == "__main__":
    main()
