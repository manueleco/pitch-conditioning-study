from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PitchLevel:
    value: float
    label: str
    display: str


def load_pitch_levels(config_path: Path) -> list[PitchLevel]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    levels = []
    for item in data["levels"]:
        levels.append(
            PitchLevel(
                value=float(item["value"]),
                label=str(item["label"]),
                display=str(item.get("display", item["value"])),
            )
        )
    return levels


def quantize_pitch(value: float, levels: list[PitchLevel]) -> PitchLevel:
    return min(levels, key=lambda level: (abs(level.value - value), abs(level.value), level.value))


def label_to_value_map(levels: list[PitchLevel]) -> dict[str, float]:
    return {level.label: level.value for level in levels}


def value_to_label_map(levels: list[PitchLevel]) -> dict[float, str]:
    return {level.value: level.label for level in levels}

