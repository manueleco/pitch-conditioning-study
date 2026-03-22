#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import torch
import zlib
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.rt import generate_offline, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera ejemplos offline a partir de los modelos entrenados."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "generation_examples.json",
        help="Configuracion de ejemplos a generar.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Directorio base de modelos.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "audio_examples",
        help="Directorio de salida de ejemplos de audio.",
    )
    parser.add_argument(
        "--continuous-model",
        type=str,
        default="two_bar_continuous_small",
        help="Nombre del directorio del modelo continuo.",
    )
    parser.add_argument(
        "--categorical-model",
        type=str,
        default="two_bar_categorical_small",
        help="Nombre del directorio del modelo categorico.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Duracion en segundos para los ejemplos generados.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperatura para hard cascade en inferencia.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Top-k para hard cascade en inferencia.",
    )
    parser.add_argument(
        "--hard-sample-mode",
        type=str,
        default=None,
        help="Modo de muestreo hard en inferencia: argmax, gumbel o sample.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=7,
        help="Semilla base para reiniciar el estado antes de cada ejemplo.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def categorical_params(model_dir: Path, label: str) -> dict[str, float]:
    config = load_json(model_dir / "conditioning_config.json")
    params = {}
    for name in config["feature_names"]:
        if not name.startswith("pitch_shift_"):
            continue
        params[name] = 1.0 if name == f"pitch_shift_{label}" else 0.0
    return params


def example_seed(base_seed: int, key: str) -> int:
    checksum = zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF
    return (int(base_seed) + checksum) % 0x7FFFFFFF


def seed_key_for_item(item: dict, categorical: bool) -> str:
    if categorical:
        return str(item["label"])

    name = str(item["name"])
    if name.startswith("continuous_interp_"):
        return name.replace("continuous_interp_", "", 1)
    if name.startswith("continuous_"):
        return name.replace("continuous_", "", 1)
    return name


def generate_set(
    model_dir: Path,
    out_dir: Path,
    duration_sec: float,
    examples: list[dict],
    categorical: bool,
    temperature: float | None = None,
    top_n: int | None = None,
    hard_sample_mode: str | None = None,
    seed_base: int = 7,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    rnngen, _, conditioning_config, _ = load_model(
        model_dir=str(model_dir),
        override_hard_sample_mode=hard_sample_mode,
        override_temperature=temperature,
        override_top_n=top_n,
    )

    for item in examples:
        if categorical:
            params = categorical_params(model_dir, item["label"])
            file_stem = item["name"]
        else:
            params = item["params"]
            file_stem = item["name"]

        seed_key = seed_key_for_item(item, categorical=categorical)
        seed_value = example_seed(seed_base, seed_key)
        torch.manual_seed(seed_value)
        rnngen.reset_state(seed=seed_value)

        audio = generate_offline(
            rnngen=rnngen,
            conditioning_config=conditioning_config,
            duration=float(duration_sec),
            param_values=params,
        )

        wav_path = out_dir / f"{file_stem}.wav"
        sf.write(wav_path, audio, 24000)
        generated.append(
            {
                "file": wav_path.name,
                "params": params,
                "duration_sec": duration_sec,
                "seed": seed_value,
            }
        )
        print(f"Ejemplo generado: {wav_path}")

    return generated


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    models_dir = args.models_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    duration_sec = float(args.duration_sec) if args.duration_sec is not None else float(cfg["duration_sec"])

    summary = {}

    continuous_model = models_dir / args.continuous_model
    if continuous_model.exists():
        summary["continuous"] = generate_set(
            model_dir=continuous_model,
            out_dir=results_dir / "continuous",
            duration_sec=duration_sec,
            examples=cfg["continuous_examples"],
            categorical=False,
            hard_sample_mode=args.hard_sample_mode,
            seed_base=args.seed_base,
            temperature=args.temperature,
            top_n=args.top_n,
        )

    categorical_model = models_dir / args.categorical_model
    if categorical_model.exists():
        summary["categorical"] = generate_set(
            model_dir=categorical_model,
            out_dir=results_dir / "categorical",
            duration_sec=duration_sec,
            examples=cfg["categorical_examples"],
            categorical=True,
            hard_sample_mode=args.hard_sample_mode,
            seed_base=args.seed_base,
            temperature=args.temperature,
            top_n=args.top_n,
        )

    summary_path = results_dir / "generated_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Resumen guardado en {summary_path}")


if __name__ == "__main__":
    main()
