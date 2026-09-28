# Resultados

## Métricas de validación

| métrica | Persistencia | Variante: DMEL + caudal | DMEL (paper) | Ablación: asignación invertida | Ablación: sin MRS |
|---|---|---|---|---|---|
| NSE_mediana | 0.525 | 0.662 | 0.624 | 0.646 | 0.665 |
| KGE_mediana | 0.653 | 0.670 | 0.666 | 0.678 | 0.647 |
| RMSE | 0.138 | 0.101 | 0.102 | 0.102 | 0.101 |
| MAE | 0.024 | 0.023 | 0.023 | 0.023 | 0.024 |
| NSE_global | 0.355 | 0.651 | 0.646 | 0.645 | 0.654 |
| cuencas_NSE<0 | 75 | 44 | 56 | 38 | 45 |
| EP_pico_mediana_% | -38.630 | -24.987 | -24.937 | -30.961 | -25.672 |
| EP_pico_abs_mediana_% | 38.794 | 34.501 | 36.777 | 37.428 | 37.122 |
| desfase_pico_abs_mediana_h | 18 | 8 | 8 | 9 | 8 |
| NSE_1h | 0.998 | 0.978 | 0.944 | 0.957 | 0.978 |
| NSE_6h | 0.957 | 0.951 | 0.930 | 0.918 | 0.949 |
| NSE_12h | 0.841 | 0.868 | 0.857 | 0.856 | 0.859 |
| NSE_24h | 0.581 | 0.663 | 0.642 | 0.668 | 0.669 |
| NSE_48h | 0.223 | 0.443 | 0.412 | 0.408 | 0.429 |

## Métricas de test

| métrica | Persistencia | Variante: DMEL + caudal | DMEL (paper) | Ablación: asignación invertida | Ablación: sin MRS |
|---|---|---|---|---|---|
| NSE_mediana | 0.492 | 0.624 | 0.614 | 0.620 | 0.629 |
| KGE_mediana | 0.626 | 0.693 | 0.655 | 0.672 | 0.641 |
| RMSE | 0.119 | 0.100 | 0.101 | 0.101 | 0.100 |
| MAE | 0.025 | 0.024 | 0.025 | 0.024 | 0.025 |
| NSE_global | 0.496 | 0.639 | 0.634 | 0.636 | 0.642 |
| cuencas_NSE<0 | 74 | 39 | 37 | 26 | 36 |
| EP_pico_mediana_% | -48.849 | -31.088 | -33.618 | -40.037 | -33.946 |
| EP_pico_abs_mediana_% | 48.908 | 36.398 | 41.268 | 44.779 | 40.286 |
| desfase_pico_abs_mediana_h | 19 | 7 | 8 | 8 | 7.500 |
| NSE_1h | 0.998 | 0.984 | 0.943 | 0.958 | 0.984 |
| NSE_6h | 0.940 | 0.948 | 0.907 | 0.919 | 0.943 |
| NSE_12h | 0.821 | 0.860 | 0.840 | 0.853 | 0.857 |
| NSE_24h | 0.538 | 0.672 | 0.649 | 0.660 | 0.656 |
| NSE_48h | 0.150 | 0.412 | 0.403 | 0.399 | 0.413 |

## Error por régimen de caudal (test, %)

| régimen | Persistencia | Variante: DMEL + caudal | DMEL (paper) | Ablación: asignación invertida | Ablación: sin MRS |
|---|---|---|---|---|---|
| bajo | 10.299 | 48.292 | 49.223 | 47.341 | 68.011 |
| medio | 16.011 | 29.525 | 31.444 | 28.678 | 33.675 |
| alto | 39.071 | 31.372 | 32.708 | 31.853 | 31.027 |

## Wilcoxon pareado por cuenca (test)

| comparación | mediana ΔNSE | cuencas donde gana | p-valor |
|---|---|---|---|
| DMEL (paper) vs Persistencia | 0.081 | 348/508 | 1.6e-19 |
| DMEL (paper) vs Variante: DMEL + caudal | -0.00729 | 221/508 | 2.0e-04 |
| DMEL (paper) vs Ablación: asignación invertida | -0.00549 | 224/508 | 4.5e-04 |
| DMEL (paper) vs Ablación: sin MRS | -0.00551 | 231/508 | 1.8e-03 |

## Entrenamiento

| corrida | parámetros (M) | épocas entrenadas | mejor época | NSE* val (mejor) | NSE val mediana (mejor) | min por época | horas totales |
|---|---|---|---|---|---|---|---|
| Variante: DMEL + caudal | 3.678 | 100 | 11 | 0.376 | 0.662 | 2.000 | 3.334 |
| DMEL (paper) | 3.677 | 100 | 11 | 0.399 | 0.618 | 1.999 | 3.332 |
| Ablación: asignación invertida | 3.675 | 100 | 5 | 0.389 | 0.645 | 1.896 | 3.160 |
| Ablación: sin MRS | 3.671 | 100 | 7 | 0.378 | 0.665 | 1.916 | 3.193 |

## MRS: entropía por componente

| componente | presente (%) | SE mediana | voto SE > SE original |
|---|---|---|---|
| IMF1 | 97.488 | 0.881 | 0.939 |
| IMF2 | 97.475 | 0.476 | 0.864 |
| IMF3 | 97.213 | 0.201 | 0.772 |
| IMF4 | 95.537 | 0.152 | 0.809 |
| IMF5 | 85.325 | 0.132 | 0.651 |
| IMF6 | 44.550 | 0.088 | 0.291 |
| residuo | 97.488 | 0.024 | 0.139 |

## MRS: grupos

| RIMF | componentes | SE mediana | voto | rama |
|---|---|---|---|---|
| RIMF1 | IMF1 | 0.881 | 0.939 | alta frecuencia → Informer |
| RIMF2 | IMF2 | 0.476 | 0.864 | alta frecuencia → Informer |
| RIMF3 | IMF3 | 0.201 | 0.772 | alta frecuencia → Informer |
| RIMF4 | IMF4 | 0.152 | 0.809 | alta frecuencia → Informer |
| RIMF5 | IMF5 | 0.132 | 0.651 | alta frecuencia → Informer |
| RIMF6 | IMF6 + residuo | 0.028 | 0.162 | baja frecuencia → LSTM |

## Fuga de información al descomponer también el futuro (validación)

| componente | descomponiendo también el futuro | control (otro ruido) |
|---|---|---|
| IMF1 | 0.00453 | 0.00429 |
| IMF2 | 0.00707 | 0.00608 |
| IMF3 | 0.015 | 0.00703 |
| IMF4 | 0.046 | 0.014 |
| IMF5 | 0.120 | 0.021 |
| IMF6 | 0.025 | 0.00464 |
| residuo | 0.179 | 0.031 |
