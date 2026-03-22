#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VALID_STEPS = ("normalize", "encode", "sidecars", "hf")


def ensure_raw_parameters(dataset_root: Path) -> None:
    root_parameters = dataset_root / "parameters.json"
    raw_dir = dataset_root / "raw"
    raw_parameters = raw_dir / "parameters.json"

    if not root_parameters.exists() or not raw_dir.exists():
        return

    if raw_parameters.exists():
        return

    shutil.copy2(root_parameters, raw_parameters)
    print(f"parameters.json copiado a {raw_parameters}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline de preparacion de datos sobre un dataset local."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Ruta al dataset, con carpetas raw, normalized, tokens y hf_dataset.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=list(VALID_STEPS),
        choices=VALID_STEPS,
        help="Pasos a ejecutar en orden.",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="RMS objetivo para la normalizacion.",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=250,
        help="Tamano de ventana para la normalizacion RMS.",
    )
    parser.add_argument(
        "--no-rms-normalization",
        action="store_true",
        help="Solo remuestrea a 24 kHz sin normalizacion RMS.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=6.0,
        help="Bandwidth de EnCodec.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dispositivo para EnCodec. Por defecto se usa cpu.",
    )
    parser.add_argument(
        "--overwrite-encodec",
        action="store_true",
        help="Sobrescribe tokens .ecdc ya existentes.",
    )
    parser.add_argument(
        "--materialize-mode",
        type=str,
        default="link",
        choices=("link", "copy", "none"),
        help="Modo de materializacion para el hf_dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    print(f"Dataset: {dataset_root}")
    print(f"Pasos: {args.steps}")

    for step in args.steps:
        print(f"\n=== Ejecutando {step} ===")

        if step == "normalize":
            from dataprep.step_1_normalization import quick_normalize

            quick_normalize(
                str(dataset_root),
                target_rms=args.target_rms,
                window_ms=args.window_ms,
                apply_rms_normalization=not args.no_rms_normalization,
            )
        elif step == "encode":
            from dataprep.step_2_encodec import quick_encode

            quick_encode(
                str(dataset_root),
                bandwidth=args.bandwidth,
                device=args.device,
                overwrite=args.overwrite_encodec,
            )
        elif step == "sidecars":
            from dataprep.step_3_sidecars import quick_create_sidecars

            ensure_raw_parameters(dataset_root)
            quick_create_sidecars(str(dataset_root))
        elif step == "hf":
            from dataprep.step_4_HF import quick_create_dataset

            ensure_raw_parameters(dataset_root)
            quick_create_dataset(
                str(dataset_root),
                materialize_mode=args.materialize_mode,
            )

    print("\nPipeline completado")


if __name__ == "__main__":
    main()
