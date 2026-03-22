# data

Esta carpeta se reserva para datasets derivados y artefactos intermedios del experimento.

## Convencion prevista

- `data/two_bar_pitch_source/`
- `data/two_bar_continuous_pitch/`
- `data/two_bar_categorical_pitch/`

Cada variante final deberia contener como minimo:

- `raw/`
- `normalized/`
- `tokens/`
- `hf_dataset/`
- `parameters.json`
- `summary.json`

## Notas

- Los audios del dataset si se versionan para la entrega
- El punto de partida actual es `data/two_bar_pitch_source/`
- La generacion de variantes se hace con `scripts/build_pitch_conditioning_variants.py`
