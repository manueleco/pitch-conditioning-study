# pitch-conditioning-study

Estudio sobre condicionamiento de pitch en generacion de audio con EnCodec y RNeNcodec.

## Objetivo

Este repositorio concentra el experimento principal del proyecto final de `Assignments Pt2`:

- comparar `pitch_shift` como condicionamiento continuo
- comparar `pitch_shift` como condicionamiento categorico con codificacion one hot
- mantener la misma base de audio y la misma arquitectura para aislar el efecto de la representacion del parametro
- evaluar si la representacion continua o la categorica ofrece un control mas claro, estable y util sobre un parametro que musicalmente es discreto

## Que queremos demostrar

El objetivo principal no es solo bajar la `loss`. La pregunta central es si el modelo responde de forma coherente cuando pedimos distintos niveles de pitch. Para eso interesa mostrar:

- barridos discretos comparables en `[-1.0, -0.5, 0.0, 0.5, 1.0]`
- puntos intermedios solo en la version continua como `0.25` y `0.75`
- seguimiento simple de pitch para comprobar monotonia del control
- escucha comparativa para detectar claridad de control y artefactos

## Alcance actual

La version actual del estudio ya no parte del recorte largo de 25.5 s. El dataset principal se reconstruye a partir de tomas originales grabadas en Logic Pro dentro de un home studio:

- `../classnn-manueleco`
- `../dataset-creation-llamita`
- `../encodec-manueleco`

La estrategia adoptada es:

- detectar el primer tramo no silencioso de cada toma original
- recortar una ventana corta de `1.9 s` aproximando los dos primeros compases
- generar cinco niveles fijos de pitch `[-1.0, -0.5, 0.0, 0.5, 1.0]`
- construir dos variantes finales a partir del mismo audio: continua y categorica
- excluir `Fur_elise_57.wav`, `Fur_elise_58.wav`, `Fur_elise_59.wav` y `Fur_elise_60.wav`

Este repo copia la base de codigo necesaria de `classnn-manueleco`, conserva scripts de referencia de `dataset-creation-llamita` y versiona el dataset corto para que el profesor pueda inspeccionarlo sin pasos extra.

## Estructura

```text
pitch-conditioning-study/
├── configs/                   # configuraciones del experimento
├── data/                      # datasets derivados y sus audios
├── dataprep/                  # pipeline copiado de classnn-manueleco
├── documents/
│   ├── diagrams/              # diagramas en Mermaid y PlantUML
│   └── paper/                 # borrador del paper en LaTeX
├── ARTIFACT_GUIDE.md          # guia rapida para navegar codigo y resultados
├── official-notebook.ipynb    # notebook principal para mostrar el flujo
├── inference/                 # inferencia copiada de classnn-manueleco
├── results/                   # tablas, figuras y ejemplos de audio
├── rnencodec/                 # paquete base copiado de classnn-manueleco
├── scripts/
│   ├── build_two_bar_pitch_source.py
│   ├── build_pitch_conditioning_variants.py
│   ├── run_dataprep_pipeline.py
│   └── reference_dataset_creation/
├── src/pitch_conditioning_study/
└── training/                  # entrenamiento copiado de classnn-manueleco
```

## Flujo previsto

1. Construir el dataset base corto con `scripts/build_two_bar_pitch_source.py`
2. Derivar las variantes continua y categorica con `scripts/build_pitch_conditioning_variants.py`
3. Ejecutar el pipeline `raw`, `normalized`, `tokens` y `hf_dataset`
4. Entrenar un modelo por variante con pocas epocas y mismo regimen
5. Comparar control perceptual, estabilidad y comportamiento entre checkpoints
6. Documentar resultados y figuras en `documents/paper/`

## Estado actual

- Repo local inicializado
- Base de codigo copiada desde `classnn-manueleco`
- Scripts de referencia del dataset copiados desde `dataset-creation-llamita`
- Estructura de paper y diagramas creada
- Documento de seguimiento en `PROJECT_STEPS.md`
- Notebook oficial creado en `official-notebook.ipynb`
- Dataset corto base creado en `data/two_bar_pitch_source`
- Variantes finales creadas en `data/two_bar_continuous_pitch` y `data/two_bar_categorical_pitch`
- Dataset de entrenamiento equilibrado por nivel de pitch con `150` clips y cerca de `4.75` minutos totales
- Carpeta `showcase/` creada con `26` tomas base no vistas para demostraciones posteriores
- Pipeline completo `normalized`, `tokens` y `hf_dataset` completado para ambas variantes
- Entrenamientos largos completados con early stopping en `models/`
- Notebook principal ejecutado y guardado con outputs reales en `official-notebook.ipynb`
- Ejemplos de audio offline regenerados desde los mejores checkpoints en `results/audio_examples/`
- Sanity check de pitch actualizado en `results/tables/`

## Resultados actuales

- modelo continuo: mejor `validation loss` `46.3799` en epoch `23`
- modelo categorico: mejor `validation loss` `46.0001` en epoch `14`
- ambos modelos mantienen monotonia en el barrido discreto de pitch
- la version continua conserva un punto intermedio razonable en `0.25`, pero el ejemplo `0.75` se pasa de largo frente a `1.0`
- la version categorica sigue una progresion mas regular en los niveles vistos, a costa de no ofrecer interpolacion natural

## Fine Tuning Adicional

Se ejecuto un segundo experimento corto reanudando desde los mejores modelos largos, con `learning_rate=0.0008`, `TF_schedule=[100, 0]` y early stopping.

- continuo fino: mejor `validation loss` `45.6274` en epoch `29`
- categorico fino: mejor `validation loss` `44.9585` en epoch `19`

La mejora en `validation loss` no se tradujo por igual en el comportamiento de generacion:

- el continuo fino mejoro en error medio de pitch discreto y en estabilidad
- el categorico fino empeoro mucho en monotonia y control de pitch durante inferencia

Por eso el resultado principal del paper sigue apoyandose en el entrenamiento largo base y usa el fine tuning como evidencia de que una `validation loss` mas baja no garantiza mejor control generativo.

## Barrido de inferencia

Tras corregir la generacion offline para reiniciar el estado interno del modelo en cada audio y fijar semillas, se ejecuto un barrido de inferencia en `results/sweeps/`.

- `baseline_sample_3s` y `matched_sample_1_9s` mantienen ambos modelos monotonos, con mejoras modestas
- `cool_sample_1_9s` mejora claramente el control continuo y consigue interpolacion monotona completa, pero rompe la monotonia del categorico
- `argmax_1_9s` no es usable en este setup y dispara el error de pitch
- `balanced_sample_1_9s` es el compromiso mas util:
  continuo `pitch_mae_discrete=0.1614`, `pitch_stability_mean=0.6084`
  categorico `pitch_mae_discrete=0.2632`, monotonia discreta `True`

Para notebook, paper y demo oral, el perfil de referencia principal es `balanced_sample_1_9s`.
El perfil `cool_sample_1_9s` se conserva como caso secundario para mostrar la mejor interpolacion del continuo.

El resumen agregado del barrido esta en `results/sweeps/summary.json`.

## Notas de trabajo

- El documento vivo del proyecto esta en `PROJECT_STEPS.md`
- Los datasets derivados viven en `data/` y sus audios si se versionan
- Los diagramas se guardan en `documents/diagrams/`
- El paper se redacta en `documents/paper/`
- El notebook principal para mostrar el proyecto es `official-notebook.ipynb`
- La guia de artefactos publicos esta en `ARTIFACT_GUIDE.md`
- La guia de escucha esta en `results/LISTENING_GUIDE.md`
- El archivo `inner-notes.md` guarda contexto operativo interno y no se sube a git
- Los audios generados para comparacion estan en `results/audio_examples/`
