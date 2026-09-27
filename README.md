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
D=$(sbatch --parsable hpc/decompose.slurm)          # CEEMDAN + MRS en CPU (~15 min)
sbatch --dependency=afterok:$D hpc/train.slurm      # dmel, dmel_swap, hybrid_raw en la A100 + evaluación
```

Salidas: `artifacts/` (componentes, `mrs.json`, tabla y figura de entropías), `runs/<corrida>/`
(checkpoint, historia de entrenamiento, predicciones), `predictions/<corrida>_test.csv` (formato
`Id, q_01..q_48` en mm/h) y `results/` (tablas y figuras). `test_targets.csv` solo lo lee
`evaluate.py` para el reporte final; la selección de modelos usa únicamente la validación.

## Resultados

*Pendiente: se completa con `results/summary.md` al terminar los entrenamientos en KHIPU.*

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
