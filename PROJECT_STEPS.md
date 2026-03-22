# PROJECT_STEPS

## Contexto general

Proyecto iniciado el **22 de marzo de 2026** para convertir el trabajo previo de `classnn-manueleco`, `encodec-manueleco` y `dataset-creation-llamita` en un estudio mas compacto y presentable.

## Pregunta principal

¿Cambia la calidad del control de pitch cuando el parametro `pitch_shift` se representa como valor continuo frente a una representacion categorica con codificacion one hot?

## Hipotesis de trabajo

- La version categorica puede dar un control mas claro y mas estable en niveles discretos
- La version continua puede resultar mas flexible, pero potencialmente menos precisa para cambios puntuales
- Si ambos modelos comparten arquitectura, datos base y regimen de entrenamiento, la diferencia principal deberia venir de la representacion del condicionamiento

## Repos de origen

- `../classnn-manueleco`
- `../dataset-creation-llamita`
- `../encodec-manueleco`

## Decisiones tomadas

- [x] Crear un repo nuevo y limpio para el estudio
- [x] Usar `pitch-conditioning-study` como nombre del repositorio
- [x] Mantener `documents/` como carpeta central para paper y diagramas
- [x] Copiar la base minima de codigo desde `classnn-manueleco`
- [x] Copiar scripts de referencia desde `dataset-creation-llamita`
- [x] No duplicar de inicio el dataset binario completo dentro del repo
- [x] Dejar un script para construir variantes `continuous` y `categorical`
- [x] Reconstruir el dataset a partir de tomas cortas de dos compases
- [x] Prescindir del augmentation viejo y generar un dataset nuevo controlado
- [x] Dejar una carpeta `showcase` con tomas no vistas para demostracion
- [x] Crear un notebook oficial del flujo principal

## Hecho

- [x] Crear estructura base del repositorio
- [x] Inicializar el repo local con git
- [x] Copiar `rnencodec/`, `dataprep/`, `training/` e `inference/`
- [x] Crear `README.md`
- [x] Crear este documento de seguimiento
- [x] Crear carpeta `documents/paper/`
- [x] Crear carpeta `documents/diagrams/`
- [x] Crear archivos iniciales para Mermaid y PlantUML
- [x] Crear borrador inicial del paper en LaTeX
- [x] Crear configuracion inicial de niveles de pitch
- [x] Crear script para generar datasets derivados del experimento
- [x] Crear script para construir el dataset corto a partir de tomas originales grabadas en Logic Pro dentro de un home studio
- [x] Crear el dataset base `two_bar_pitch_source`
- [x] Crear las variantes `two_bar_continuous_pitch` y `two_bar_categorical_pitch`
- [x] Conseguir una particion balanceada por niveles de pitch
- [x] Crear `official-notebook.ipynb`
- [x] Crear `scripts/run_dataprep_pipeline.py` para encadenar `step_1..4`
- [x] Crear `inner-notes.md` para contexto operativo fuera de git

## Pendiente inmediato

- [x] Ejecutar `scripts/build_two_bar_pitch_source.py`
- [x] Ejecutar `scripts/build_pitch_conditioning_variants.py`
- [x] Verificar distribucion final de niveles por split
- [x] Ejecutar `step_1_normalization.py` sobre la variante continua
- [x] Ejecutar `step_2_encodec.py` sobre la variante continua
- [x] Ejecutar `step_1_normalization.py` sobre la variante categorica
- [x] Ejecutar `step_2_encodec.py` sobre la variante categorica
- [x] Ejecutar `step_3_sidecars.py` sobre cada variante
- [x] Ejecutar `step_4_HF.py` sobre cada variante
- [x] Definir regimen de entrenamiento corto para ambos modelos
- [x] Entrenar la version continua
- [x] Entrenar la version categorica
- [x] Guardar audios de ejemplo para comparacion
- [x] Ejecutar entrenamiento largo con early stopping
- [x] Regenerar audios desde los mejores checkpoints
- [x] Ejecutar sanity check de monotonia e interpolacion
- [x] Ejecutar fine tuning con learning rate mas bajo
- [x] Medir control de pitch y ataque inicial con metricas objetivas
- [x] Ejecutar barrido de inferencia con semillas fijas y reinicio de estado por audio
- [x] Preparar tabla comparativa para el paper

## Pendiente de documentacion

- [ ] Sustituir el borrador de LaTeX por el template oficial de ISMIR 2025
- [x] Redactar abstract
- [x] Redactar metodologia
- [x] Redactar resultados
- [x] Redactar limitaciones
- [ ] Insertar diagramas en el paper
- [ ] Añadir capturas o tablas extraidas del notebook oficial

## Riesgos actuales

- El material base no es una textura pura, sino una pieza o fragmentos de pieza
- El presupuesto de computo es limitado
- Puede hacer falta elegir checkpoints tempranos si aparece sobreajuste
- La comparacion solo sera limpia si ambos modelos usan los mismos niveles discretizados de pitch
- El modelo seguira entrenandose sobre material melodico y no sobre una textura pura
- Los audios `showcase` no deben mezclarse con train o validation

## Hallazgos tempranos

- Se adopto una ventana corta de `1.9 s` con `50 ms` de pre roll tras el primer tramo no silencioso
- El dataset base `two_bar_pitch_source` usa `20` tomas base en train, `5` en validation y `5` en test
- La duracion total de `train + validation + test` es de aproximadamente `4.75` minutos
- La distribucion por pitch ya esta balanceada:
  `train`: 20 clips por nivel
  `validation`: 5 clips por nivel
  `test`: 5 clips por nivel
- La carpeta `showcase` contiene `26` tomas base no vistas, tambien expandida a cinco niveles de pitch
- Las variantes finales `two_bar_continuous_pitch` y `two_bar_categorical_pitch` comparten exactamente el mismo audio
- `official-notebook.ipynb` ya resume construccion, verificacion y escucha de ejemplos
- `data/two_bar_continuous_pitch` y `data/two_bar_categorical_pitch` ya tienen `normalized/`, `tokens/` y `hf_dataset/` completos
- Entrenamiento continuo completado en `8` epocas con `validation loss` final `49.8238`
- Entrenamiento categorico completado en `8` epocas con `validation loss` final `48.5274`
- `results/audio_examples/` contiene barridos discretos para ambos modelos y dos puntos intermedios para la version continua
- Entrenamiento largo continuo completado con early stopping en epoch `31`, mejor epoch `23` y mejor `validation loss` `46.3799`
- Entrenamiento largo categorico completado con early stopping en epoch `22`, mejor epoch `14` y mejor `validation loss` `46.0001`
- La inferencia se corrigio para usar `last_checkpoint.pt`, que corresponde al mejor checkpoint restaurado
- Ambos modelos largos mantienen monotonia en el barrido discreto
- La interpolacion continua `up_0_25` cae entre sus vecinos, pero `up_0_75` supera el valor de `up_1`
- `official-notebook.ipynb` ya fue ejecutado y guardado con outputs
- Fine tuning continuo: mejor `validation loss` `45.6274` en epoch `29`
- Fine tuning categorico: mejor `validation loss` `44.9585` en epoch `19`
- El fine tuning mejoro el error discreto de pitch del continuo de `0.2641` a `0.1691` semitonos
- El fine tuning categorico empeoro el error discreto de pitch de `0.1381` a `1.7333` semitonos y rompio la monotonia
- Las metricas objetivas y figuras se guardan en `results/tables/objective_summary*.json` y `results/figures*/`
- El ataque inicial queda cuantificado con ratios onset/cuerpo y con figuras STFT, en lugar de depender solo de escucha subjetiva
- Se corrigio la generacion offline para reiniciar el estado interno del generador en cada audio y fijar semillas reproducibles
- El barrido de inferencia principal vive en `configs/inference_sweep.json`, `scripts/run_inference_sweep.py` y `results/sweeps/summary.json`
- `argmax` no funciono como mejora en este setup y produjo errores extremos de pitch
- `balanced_sample_1_9s` queda fijado como perfil de referencia principal para notebook, paper y demo oral
- `cool_sample_1_9s` queda como perfil secundario para ilustrar la mejor interpolacion del modelo continuo
- El perfil `cool_sample_1_9s` mejoro mucho el control continuo:
  continuo `pitch_mae_discrete=0.1306`
  interpolacion monotona `True`
  `up_0_25` y `up_0_75` entre vecinos `True`
  pero rompio el categorico con `pitch_mae_discrete=1.9723` y monotonia `False`
- El perfil `balanced_sample_1_9s` ofrecio el mejor compromiso global:
  continuo `pitch_mae_discrete=0.1614`, `pitch_stability_mean=0.6084`, monotonia `True`
  categorico `pitch_mae_discrete=0.2632`, monotonia `True`
  sin llegar al desastre de `argmax` ni a la rotura del categorico en `cool_sample_1_9s`

## Criterios minimos de cierre

- Dos datasets derivados comparables
- Dos entrenamientos ejecutados
- Un conjunto de ejemplos de inferencia por variante
- Una tabla o figura que muestre diferencias claras
- Un paper corto con estructura coherente

## Mejoras a futuro

- Añadir un segundo parametro experimental orientado a textura
- Construir un dataset mas corto y mas repetitivo basado en el trino
- Probar reverberacion o espacio como segundo control
- Añadir evaluacion perceptual informal con escucha comparativa
- Preparar una version publica con audios y demo reproducible
