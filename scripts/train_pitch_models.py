#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.loop import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena los modelos continuo y categorico con la misma configuracion base."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_small.json",
        help="Archivo JSON con la configuracion de entrenamiento.",
    )
    parser.add_argument(
        "--continuous-dataset",
        type=Path,
        default=ROOT / "data" / "two_bar_continuous_pitch",
        help="Dataset continuo.",
    )
    parser.add_argument(
        "--categorical-dataset",
        type=Path,
        default=ROOT / "data" / "two_bar_categorical_pitch",
        help="Dataset categorico.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Directorio de salida de modelos.",
    )
    parser.add_argument(
        "--target",
        choices=("both", "continuous", "categorical"),
        default="both",
        help="Modelo o modelos a entrenar.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_resume_dir(resume_value: str | None, models_dir: Path) -> Path | None:
    if not resume_value:
        return None
    resume_path = Path(resume_value)
    if not resume_path.is_absolute():
        resume_path = models_dir / resume_path
    return resume_path.resolve()


def prepare_resume_copy(resume_dir: Path | None, output_dir: Path) -> Path | None:
    if resume_dir is None:
        return None
    if output_dir.exists():
        return output_dir
    shutil.copytree(resume_dir, output_dir)
    print(f"📁 Copia inicial para fine tuning: {resume_dir} -> {output_dir}")
    return output_dir


def run_training(
    dataset_path: Path,
    model_name: str,
    cfg: dict,
    models_dir: Path,
    resume_dir: Path | None = None,
) -> dict:
    device = select_device(cfg.get("device", "cpu"))
    output_dir = models_dir / model_name
    prepared_resume_dir = prepare_resume_copy(resume_dir, output_dir)

    print(f"\nIniciando entrenamiento: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    if prepared_resume_dir is not None:
        print(f"Resume desde: {prepared_resume_dir}")

    result = train_model(
        dataset_path=str(dataset_path),
        model_output_path=str(output_dir),
        num_epochs=int(cfg["num_epochs"]),
        batch_size=int(cfg["batch_size"]),
        sequence_length=int(cfg["sequence_length"]),
        batches_per_epoch=int(cfg["batches_per_epoch"]),
        learning_rate=float(cfg["learning_rate"]),
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        cascade_mode=str(cfg["cascade_mode"]),
        temperature=float(cfg["temperature"]),
        top_n=int(cfg["top_n"]) if cfg.get("top_n") is not None else None,
        tau_soft=float(cfg["tau_soft"]),
        save_interval=int(cfg["save_interval"]),
        device=device,
        train_splits=cfg.get("train_splits"),
        val_splits=cfg.get("val_splits"),
        use_tensorboard=bool(cfg.get("use_tensorboard", True)),
        use_tqdm=bool(cfg.get("use_tqdm", False)),
        val_interval=int(cfg.get("val_interval", 1)),
        early_stopping_patience=(
            int(cfg["early_stopping_patience"])
            if cfg.get("early_stopping_patience") is not None
            else None
        ),
        early_stopping_min_delta=float(cfg.get("early_stopping_min_delta", 0.0)),
        restore_best_checkpoint=bool(cfg.get("restore_best_checkpoint", True)),
        files_per_sequence=int(cfg.get("files_per_sequence", 4)),
        add_noise=bool(cfg.get("add_noise", False)),
        noise_weight=float(cfg.get("noise_weight", 0.0)),
        dropout=float(cfg.get("dropout", 0.1)),
        TF_schedule=cfg.get("TF_schedule"),
        quantizer_weights=cfg.get("quantizer_weights"),
        simulate_parallel=bool(cfg.get("simulate_parallel", False)),
        resume_checkpoint=str(prepared_resume_dir) if prepared_resume_dir is not None else None,
    )
    return result


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 7))
    random.seed(seed)
    torch.manual_seed(seed)

    targets = []
    if args.target in ("both", "continuous"):
        continuous_name = str(cfg.get("continuous_model_name", "two_bar_continuous_small"))
        targets.append(("continuous", args.continuous_dataset.resolve(), continuous_name))
    if args.target in ("both", "categorical"):
        categorical_name = str(cfg.get("categorical_model_name", "two_bar_categorical_small"))
        targets.append(("categorical", args.categorical_dataset.resolve(), categorical_name))

    summary = {}
    for tag, dataset_path, model_name in targets:
        resume_key = f"{tag}_resume_from"
        resume_dir = resolve_resume_dir(cfg.get(resume_key), models_dir=models_dir)
        summary[tag] = run_training(
            dataset_path,
            model_name,
            cfg,
            models_dir=models_dir,
            resume_dir=resume_dir,
        )

    summary_path = models_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"\nResumen guardado en {summary_path}")


if __name__ == "__main__":
    main()
