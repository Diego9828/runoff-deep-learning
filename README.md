# DMEL para pronóstico horario de caudal — Laboratorio 2

Implementación propia y adaptación de **DMEL** (*Dual-Modal Ensemble Learning*; Wang et al., 2026,
*Applied Soft Computing* 192:114792) al dataset del laboratorio: a partir de **336 h de 12 variables**
(caudal + 11 meteorológicas) de una cuenca se pronostican las **48 h siguientes de caudal** (mm/h),
con 508 cuencas.

## Método

DMEL tiene dos piezas:

1. **MRS (Modal Recognition Strategy)**: se descompone el caudal con **CEEMDAN** (ruido 0.2, 100
   realizaciones), se mide la complejidad de cada componente con **sample entropy** (m = 2, r = 0.2·std)
   y se reagrupan los componentes de entropía parecida (RIMFs). Los RIMFs con entropía mayor que la de la
   serie original son de **alta frecuencia (HF)**; el resto, de **baja frecuencia (LF)**.
2. **Ensamble dual**: un **Informer** pronostica con los RIMFs de alta frecuencia y un **LSTM** con los de
   baja; la predicción es la **suma** de ambas ramas (Ec. 14 del paper). Como en la versión multivariable
   del paper (Sec. 5.6), las covariables meteorológicas entran a las dos ramas.

```
historia 336 h ──► caudal ──► CEEMDAN (por ventana) ──► SE + votos ──► RIMF HF ─┐
                │                                                  └─► RIMF LF ─┼─► Informer(HF + covariables) ─┐
                └► 11 covariables + 2 estáticos de la cuenca ──────────────────┘   LSTM(LF + covariables) ─────┴─► suma ─► 48 h
```

### Qué se implementó (todo desde cero, en `src/dmel/`)

| Archivo | Contenido |
|---|---|
| `ceemdan.py` | CEEMDAN mejorado (Colominas et al., 2014) en numba: sifting, spline cúbica natural, banco de ruido |
| `mrs.py` | Sample entropy (Richman y Moorman, 2000), votos por componente y regla de agrupación del MRS |
| `informer.py` | Informer: atención ProbSparse, distilling, decoder generativo |
| `models.py` | LSTM, ensamble dual y variantes de la ablación |
| `data.py` | Lectura de los `.h5`, escalado por cuenca, construcción y normalización de entradas |
| `metrics.py` | NSE, KGE, RMSE, MAE por cuenca y por horizonte; error de pico |
| `plots.py` | Figuras del reporte y la presentación |
| `decompose.py`, `build_mrs.py`, `train.py`, `evaluate.py` | CLIs del flujo completo |

### Adaptaciones al dataset (diferencias con el paper)

| Paper | Este trabajo | Motivo |
|---|---|---|
| CEEMDAN de la serie completa y luego división 7:1.5:1.5 | CEEMDAN **de cada ventana, solo con sus 336 h de historia** | Descomponer la serie completa usa el futuro (fuga de información; ver `results/leakage.png`) y en test solo se tiene la historia |
| Agrupación visual por estación | **Regla automática global**: voto por componente = fracción de ventanas de train con SE_k > SE(original); voto > 0.5 → RIMF HF individual; el resto se suma por bandas de SE (≤ 0.1 y > 0.1) y se re-clasifica con su propio voto | Un solo modelo para 508 cuencas necesita canales fijos. La regla reproduce la Tabla 3 del paper en sus dos estaciones (`tests/test_mrs.py`) |
| Cada modelo predice los componentes futuros | Cada rama aporta una parte del caudal futuro y se entrena **end-to-end sobre la suma** | Los componentes futuros no existen en inferencia |
| Datos diarios, entrada 7 días, salida 1/3/5/7 días | Entrada 336 h, salida directa de 48 h | Enunciado del laboratorio |
| Embedding temporal global del Informer | Se omite (solo Conv1d + codificación posicional) | El dataset no tiene fechas |
| MSE | **NSE\*** (Kratzert et al., 2019): MSE sobre y / s_b, con s_b = σ_b + 0.01 mm/h de cada cuenca | El caudal varía ~3 órdenes de magnitud entre cuencas |
| Una estación | 2 canales estáticos (log s_b, log μ_b) en todos los modelos | El escalado por cuenca borra la magnitud del caudal |
| Batch 32, lr 1e-4, 100 épocas | Batch 256, lr 3e-4 con warmup de 500 pasos, ReduceLROnPlateau, early stopping (paciencia 5, máx. 30 épocas) | ~250× más datos |
| 10 corridas + Wilcoxon entre corridas | 1 semilla + Wilcoxon pareado sobre el NSE de las 508 cuencas | Tiempo de GPU (1 GPU por usuario en KHIPU) |
| Código oficial del Informer | Un solo muestreo de claves por pasada, media correcta, media acumulada para queries causales, padding `replicate` | Evita un tensor de ~2.6 GB por capa y que la hora 336 quede pegada a la hora 1 |

Hiperparámetros como en la Tabla 1 del paper: Informer d_model 256, 8 cabezas, 2 capas de encoder,
1 de decoder, d_ff 1024, GELU; LSTM 2 × 256. CEEMDAN con 6 IMFs + residuo (336 muestras soportan ~5–6 modos).

### Experimentos

| Corrida | Rama 1 | Rama 2 | Pregunta |
|---|---|---|---|
| Persistencia | — | — | Modelo base ingenuo: último caudal observado |
| **`dmel`** | Informer(RIMFs HF) | LSTM(RIMFs LF) | Propuesta del paper |
| `dmel_swap` | LSTM(RIMFs HF) | Informer(RIMFs LF) | Ablación 1: ¿importa asignar cada frecuencia a su modelo? (paper, Sec. 5.2) |
| `hybrid_raw` | Informer(caudal crudo) | LSTM(caudal crudo) | Ablación 2: ¿aporta el MRS? Misma arquitectura sin descomposición (modelo base entrenado) |
| `dmel_q` | Informer(RIMFs HF) | LSTM(RIMFs LF + caudal crudo) | Variante propia: ¿recupera DMEL las primeras horas si también ve el caudal observado? |

## Cómo ejecutar

Requisitos: [uv](https://docs.astral.sh/uv/). Los datos van en `data/` (`train.h5`, `test.h5`,
`test_targets.csv`); `data/` no se versiona.

```bash
uv sync
uv run pytest
```

Prueba rápida local (unos minutos):

```bash
uv run dmel-decompose --split train --limit 2000 --out smoke/artifacts
uv run dmel-decompose --split test --limit 2000 --out smoke/artifacts
uv run dmel-mrs --artifacts smoke/artifacts --windows 2000
uv run dmel-train --model dmel --limit 2000 --epochs 2 --artifacts smoke/artifacts --runs smoke/runs --predictions smoke/predictions
uv run dmel-evaluate --limit 2000 --runs smoke/runs --out smoke/results
```

Flujo completo en KHIPU:

```bash
bash hpc/push.sh                                    # sube el commit actual y hace uv sync
ssh khipu
cd runoff_deep_learning
D=$(sbatch --parsable hpc/decompose.slurm)          # CEEMDAN + MRS en CPU (~25 min con 24 CPUs)
T=$(MODELS="dmel dmel_q" sbatch --parsable --dependency=afterok:$D hpc/train.slurm)
MODELS="dmel_swap hybrid_raw" sbatch --dependency=afterany:$T hpc/train.slurm   # + evaluación final
```

Con 100 épocas cada modelo toma ~3.4 h en una RTX A6000, así que van dos modelos por job (límite de 8 h).

`train.slurm` pide la A100 completa (`gpu:a100:1`). Si está ocupada, se puede usar un slice MIG de la
misma A100 sin editar el script: `sbatch --gres=gpu:a100_3g.20gb:1 ...` (20 GB alcanzan con batch 256).

Traer los resultados y generar el reporte y la presentación ([Quarto](https://quarto.org) ≥ 1.8):

```bash
bash hpc/pull.sh                                    # results/, predictions/, historias y logs
quarto render REPORT.qmd                            # REPORT.html (autocontenido)
quarto render PRESENTATION.qmd                      # PRESENTATION.html (revealjs)
```

Salidas: `artifacts/` (componentes, `mrs.json`, tabla y figura de entropías), `runs/<corrida>/`
(checkpoint, historia de entrenamiento, predicciones), `predictions/<corrida>_test.csv` (formato
`Id, q_01..q_48` en mm/h) y `results/` (tablas y figuras). `test_targets.csv` solo lo lee
`evaluate.py` para el reporte final; la selección de modelos usa únicamente la validación.

## Configuración y reproducibilidad

Todas las corridas usan los valores por defecto de los CLIs (los scripts de `hpc/` no los sobrescriben).
Cada corrida guarda sus argumentos, el número de parámetros y las estadísticas de normalización en
`runs/<corrida>/config.json`.

### Entorno

| Elemento | Valor |
|---|---|
| Python | 3.12 (`.python-version`); dependencias fijadas en `uv.lock` (`uv sync --frozen`) |
| Librerías principales | torch 2.11.0+cu126 (índice PyTorch cu126), numba 0.67, numpy 2.4, scipy 1.18, h5py 3.16 |
| CEEMDAN + MRS | KHIPU, partición `standard`, 24 CPUs (job 53335: 38 min) |
| Entrenamiento | KHIPU, partición `gpu`, NVIDIA RTX A6000 48 GB (job 53336). La A100 estaba reservada por otros usuarios; el mismo script corre en `gpu:a100:1` o en el slice `a100_3g.20gb` |
| Código ejecutado | el commit queda en `REVISION` (KHIPU) y en la primera línea de cada log de Slurm (`logs/*.out`) |

### Semillas

| Qué | Semilla | Dónde |
|---|---|---|
| Pesos, dropout y orden de los batches | `--seed 0` | `src/dmel/train.py` |
| Banco de 100 realizaciones de ruido de CEEMDAN (el mismo para train y test) | 0 | `noise_bank()` en `src/dmel/ceemdan.py` |
| Muestra de 8000 ventanas de train para el MRS | 0 | `--seed` en `src/dmel/build_mrs.py` |
| Muestra de 20 000 ventanas para la normalización | 0 | `fit_norm()` en `src/dmel/data.py` |
| Muestreo de claves de ProbSparse al predecir | 0 | `predict()` en `src/dmel/train.py` |

### Hiperparámetros

| Etapa | Parámetro | Valor | Dónde |
|---|---|---|---|
| CEEMDAN | IMFs máximos + residuo | 6 + 1 | `--imfs`, `src/dmel/decompose.py` |
| | Realizaciones de ruido | 100 (paper: 100) | `--trials` |
| | Amplitud relativa del ruido ε | 0.2 (paper: 0.2) | `--eps` |
| | Iteraciones de sifting por IMF | 10 (fijas) | `--sifts` |
| | Envolventes | spline cúbica natural; bordes por extrapolación lineal de los 2 extremos más cercanos | `envelope()`, `src/dmel/ceemdan.py` |
| Sample entropy | m, r | 2, 0.2·std (paper: igual) | `sample_entropy()`, `src/dmel/mrs.py` |
| MRS | Ventanas de train | 8000 | `--windows`, `src/dmel/build_mrs.py` |
| | Umbral de voto / banda τ | 0.5 / 0.1 | `propose_groups()` y `--tau` |
| | Resultado | HF = IMF1…IMF5 (5 RIMFs); LF = IMF6 + residuo (1 RIMF) | `artifacts/mrs.json` |
| Entradas | Escala por cuenca | s_b = σ_b + ε, ε = 0.01 mm/h (estadísticas de train) | `EPS`, `basin_scales()`, `src/dmel/data.py` |
| | Canales por rama | hf: 5 RIMFs + 11 meteorológicas + 2 estáticos = 18; lf: 1 + 11 + 2 = 14; raw: 1 + 11 + 2 = 14 | `make_inputs()` |
| | Normalización | z-score por canal con 20 000 ventanas de train | `fit_norm()` |
| Informer | d_model, cabezas, d_ff | 256, 8, 1024 (paper: igual) | `Informer.__init__`, `src/dmel/informer.py` |
| | Capas encoder / decoder | 2 / 1, con 1 distilling (336 → 168) | |
| | ProbSparse | c = 5 → u = 30 queries en el encoder, 25 en el decoder | `prob_attention()` |
| | Decoder | label_len 48 + 48 ceros; activación GELU; dropout 0.1 | |
| LSTM | Capas, unidades, dropout | 2, 256, 0.1 (paper: 2, 256); cabeza Linear(256 → 48) | `LSTMNet`, `src/dmel/models.py` |
| Modelo | Parámetros | 3.68 M (dmel) | log de entrenamiento |
| Optimización | Optimizador, lr | Adam, 3e-4 con warmup lineal de 500 pasos | `--lr`, `--warmup`, `src/dmel/train.py` |
| | Batch | 256 (993 pasos por época) | `--batch` |
| | Épocas | 100 (paper: 100), sin early stopping; se guarda la época con menor pérdida de validación | `--epochs`; `--patience` opcional |
| | Scheduler | ReduceLROnPlateau (factor 0.5, paciencia 2) | `main()`, `src/dmel/train.py` |
| | Gradiente | clip de norma 1.0; TF32 activado | `--clip` |
| | Pérdida | NSE\* = MSE(ŷ/s_b, y/s_b) | `main()` |
| Predicción | Postproceso | ŷ·s_b recortado en 0 mm/h | `main()` |
| Evaluación | Métricas | NSE y KGE por cuenca (mediana entre cuencas), RMSE y MAE agregados, NSE por horizonte | `src/dmel/metrics.py` |
| | Crecidas | ventanas cuyo pico observado supera el cuantil 0.95 de su cuenca | `peak_errors()` |
| | Regímenes | terciles del caudal observado de cada cuenca | `regimes_table()`, `src/dmel/evaluate.py` |
| | Significancia | Wilcoxon pareado sobre el NSE de las cuencas | `wilcoxon_table()` |

Para cambiar un valor sin tocar el código se pasa el argumento (por ejemplo
`dmel-train --model dmel --epochs 100`).

## Resultados

Las tablas de `results/tables/*.md` y las figuras de `results/figures/*.png` se regeneran con
`dmel-evaluate`; el reporte (`REPORT.qmd`) y la presentación (`PRESENTATION.qmd`) las incluyen.

> **Nota:** estos números son de la primera ronda, con early stopping (paciencia 5, máximo 30 épocas).
> Se están reentrenando todos los modelos, más la variante `dmel_q`, con 100 épocas completas. Las
> métricas de esta ronda quedan archivadas en `results/earlystop/`.

Test (27 983 ventanas, 508 cuencas; NSE y KGE son medianas entre cuencas):

| | Persistencia | **DMEL** | Asignación invertida | Sin MRS |
|---|---:|---:|---:|---:|
| NSE mediano | 0.492 | **0.629** | 0.629 | 0.625 |
| KGE mediano | 0.626 | 0.673 | 0.677 | 0.681 |
| RMSE (mm/h) | 0.119 | 0.101 | 0.101 | 0.100 |
| NSE a 1 h / 24 h / 48 h | 0.998 / 0.538 / 0.150 | 0.957 / 0.651 / 0.407 | 0.953 / 0.675 / 0.412 | 0.983 / 0.668 / 0.400 |
| Cuencas con NSE < 0 | 74 | 39 | 24 | 26 |
| Mejor época / épocas entrenadas | — | 11 / 16 | 5 / 10 | 5 / 10 |

- **DMEL supera a la persistencia** en 343 de 508 cuencas (Wilcoxon p = 6.2e-19), sobre todo desde ~8 h;
  en las primeras horas copiar el último valor es mejor.
- **Ablación 1 (asignación invertida)**: sin diferencia significativa (p = 0.087).
- **Ablación 2 (sin MRS)**: la misma arquitectura con el caudal crudo es levemente mejor (p = 2.6e-05).
- Descomponer la serie completa, como el paper, hace que los componentes lentos cambien 4–6× más que el
  ruido propio de CEEMDAN (`results/figures/leakage.png`): es una explicación probable de parte de las
  ganancias reportadas en el paper.
- Errores principales: crecidas provocadas por lluvia que aún no cae (no hay meteorología futura en
  inferencia), picos subestimados (~30–40 %) y caudales bajos, donde la persistencia es mucho mejor.

El análisis completo está en `REPORT.qmd` / `REPORT.html`; las predicciones de test, en
`predictions/dmel_s0_test.csv` (y una por cada ablación).

## Limitaciones

- La descomposición por ventana tiene efecto de borde: las últimas horas de cada componente, las más
  importantes para pronosticar, son las menos estables. Es el precio de no usar el futuro.
- La agrupación del MRS es global. Por cuenca podría cambiar algún componente limítrofe (IMF5/IMF6).
- Las ramas no se supervisan por separado, así que su aporte (`results/events.png`) es una
  interpretación, no un pronóstico de cada banda de frecuencia.
- Una sola semilla por modelo.

## Uso de IA generativa

Este trabajo se desarrolló con asistencia de IA generativa (Claude, de Anthropic) para escribir y
revisar código y documentación. El grupo revisó y validó las decisiones de implementación y los resultados.

## Referencias

- Wang, W.-c., Gu, M., Li, Z., Wang, Y.-y., Zhao, Y.-w. (2026). DMEL: A novel dual-modal ensemble learning architecture for multi-step runoff prediction. *Applied Soft Computing*, 192, 114792.
- Colominas, M. A., Schlotthauer, G., Torres, M. E. (2014). Improved complete ensemble EMD: A suitable tool for biomedical signal processing. *Biomedical Signal Processing and Control*, 14, 19–29.
- Zhou, H. et al. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *AAAI*.
- Richman, J. S., Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *Am. J. Physiol. Heart Circ. Physiol.*, 278(6).
- Kratzert, F. et al. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *HESS*, 23, 5089–5110.
- Fang, W. et al. (2019). Examining the applicability of different sampling techniques in the development of decomposition-based streamflow forecasting models. *Journal of Hydrology*, 568, 534–550.
