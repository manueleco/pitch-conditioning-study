#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "inference_sweep.json"
DEFAULT_RESULTS_ROOT = ROOT / "results" / "sweeps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta un barrido de inferencia y resume metricas objetivas."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Configuracion JSON del barrido.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directorio raiz para los perfiles generados.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=ROOT.parent / "classnn-manueleco" / ".venv" / "bin" / "python",
        help="Python del entorno con dependencias del proyecto.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(cmd: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("")
    print("Ejecutando:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build_generate_command(
    python_bin: Path,
    continuous_model: str,
    categorical_model: str,
    profile_dir: Path,
    profile: dict,
    seed_base: int,
) -> list[str]:
    cmd = [
        str(python_bin),
        "scripts/generate_training_examples.py",
        "--continuous-model",
        continuous_model,
        "--categorical-model",
        categorical_model,
        "--results-dir",
        str(profile_dir),
        "--duration-sec",
        str(profile["duration_sec"]),
        "--seed-base",
        str(seed_base),
    ]

    hard_sample_mode = profile.get("hard_sample_mode")
    if hard_sample_mode is not None:
        cmd.extend(["--hard-sample-mode", str(hard_sample_mode)])

    temperature = profile.get("temperature")
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])

    top_n = profile.get("top_n")
    if top_n is not None:
        cmd.extend(["--top-n", str(top_n)])

    return cmd


def summarize_profile(profile_name: str, profile: dict, profile_dir: Path) -> dict:
    objective_path = profile_dir / "tables" / "objective_summary.json"
    pitch_path = profile_dir / "tables" / "pitch_tracking_report.json"

    objective = load_json(objective_path)
    pitch = load_json(pitch_path)

    result = {
        "profile": profile_name,
        "settings": profile,
        "continuous": {},
        "categorical": {},
    }

    for model_name in ["continuous", "categorical"]:
        model_obj = objective["models"][model_name]
        model_pitch = pitch["models"][model_name]
        result[model_name] = {
            "pitch_mae_discrete_semitones": model_obj["pitch_mae_discrete_semitones"],
            "pitch_mae_all_semitones": model_obj["pitch_mae_all_semitones"],
            "pitch_stability_mean_semitones": model_obj["pitch_stability_mean_semitones"],
            "onset_rms_ratio_mean": model_obj["onset_rms_ratio_mean"],
            "onset_flatness_ratio_mean": model_obj["onset_flatness_ratio_mean"],
            "monotonic_discrete_sweep": model_pitch["monotonic_discrete_sweep"]["is_monotonic"],
            "interpolation_monotonic": model_pitch["continuous_interpolation"]["is_monotonic"],
            "up_0_25_between_neighbors": model_pitch["continuous_interpolation"]["up_0_25_between_neighbors"],
            "up_0_75_between_neighbors": model_pitch["continuous_interpolation"]["up_0_75_between_neighbors"],
        }

    return result


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    results_root = args.results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    continuous_model = str(cfg["continuous_model"])
    categorical_model = str(cfg["categorical_model"])
    seed_base = int(cfg.get("seed_base", 7))
    profiles = list(cfg["profiles"])

    summary_rows = []

    for profile in profiles:
        profile_name = str(profile["name"])
        profile_dir = results_root / profile_name
        tables_dir = profile_dir / "tables"
        figures_dir = profile_dir / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        run_command(
            build_generate_command(
                python_bin=args.python_bin,
                continuous_model=continuous_model,
                categorical_model=categorical_model,
                profile_dir=profile_dir,
                profile=profile,
                seed_base=seed_base,
            ),
            cwd=ROOT,
        )

        run_command(
            [
                str(args.python_bin),
                "scripts/evaluate_generated_pitch.py",
                "--results-dir",
                str(profile_dir),
                "--output-csv",
                str(tables_dir / "pitch_tracking_summary.csv"),
                "--output-json",
                str(tables_dir / "pitch_tracking_report.json"),
            ],
            cwd=ROOT,
        )

        run_command(
            [
                str(args.python_bin),
                "scripts/analyze_generated_audio.py",
                "--results-dir",
                str(profile_dir),
                "--output-csv",
                str(tables_dir / "objective_metrics.csv"),
                "--output-json",
                str(tables_dir / "objective_summary.json"),
                "--figures-dir",
                str(figures_dir),
            ],
            cwd=ROOT,
        )

        row = summarize_profile(profile_name, profile, profile_dir)
        summary_rows.append(row)
        print("")
        print(f"Perfil completado: {profile_name}")
        print(
            "  continuo   "
            f"mae={row['continuous']['pitch_mae_discrete_semitones']:.4f} "
            f"mono={row['continuous']['monotonic_discrete_sweep']} "
            f"onset_rms={row['continuous']['onset_rms_ratio_mean']:.4f}"
        )
        print(
            "  categorico "
            f"mae={row['categorical']['pitch_mae_discrete_semitones']:.4f} "
            f"mono={row['categorical']['monotonic_discrete_sweep']} "
            f"onset_rms={row['categorical']['onset_rms_ratio_mean']:.4f}"
        )

    summary = {
        "seed_base": seed_base,
        "continuous_model": continuous_model,
        "categorical_model": categorical_model,
        "profiles": summary_rows,
    }

    summary_path = results_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("")
    print(f"Resumen guardado en {summary_path}")


if __name__ == "__main__":
    main()
