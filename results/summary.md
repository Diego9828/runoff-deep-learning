# Resultados

## Métricas de validación

| métrica | Persistencia | DMEL (paper) | Ablación: asignación invertida | Ablación: sin MRS |
|---|---|---|---|---|
| NSE_mediana | 0.525 | 0.638 | 0.641 | 0.655 |
| KGE_mediana | 0.653 | 0.673 | 0.694 | 0.660 |
| RMSE | 0.138 | 0.103 | 0.102 | 0.102 |
| MAE | 0.024 | 0.023 | 0.024 | 0.023 |
| NSE_global | 0.355 | 0.641 | 0.650 | 0.649 |
| cuencas_NSE<0 | 75 | 52 | 38 | 40 |
| EP_pico_mediana_% | -38.630 | -23.474 | -29.744 | -25.318 |
| EP_pico_abs_mediana_% | 38.794 | 36.824 | 36.659 | 36.643 |
| desfase_pico_abs_mediana_h | 18 | 8 | 8 | 9 |
| NSE_1h | 0.998 | 0.957 | 0.949 | 0.978 |
| NSE_6h | 0.957 | 0.926 | 0.916 | 0.947 |
| NSE_12h | 0.841 | 0.840 | 0.857 | 0.872 |
| NSE_24h | 0.581 | 0.636 | 0.663 | 0.680 |
| NSE_48h | 0.223 | 0.418 | 0.414 | 0.404 |

## Métricas de test

| métrica | Persistencia | DMEL (paper) | Ablación: asignación invertida | Ablación: sin MRS |
|---|---|---|---|---|
| NSE_mediana | 0.492 | 0.629 | 0.629 | 0.625 |
| KGE_mediana | 0.626 | 0.673 | 0.677 | 0.681 |
| RMSE | 0.119 | 0.101 | 0.101 | 0.100 |
| MAE | 0.025 | 0.024 | 0.024 | 0.024 |
| NSE_global | 0.496 | 0.635 | 0.634 | 0.642 |
| cuencas_NSE<0 | 74 | 39 | 24 | 26 |
| EP_pico_mediana_% | -48.849 | -32.218 | -38.983 | -31.491 |
| EP_pico_abs_mediana_% | 48.908 | 40.427 | 43.150 | 38.623 |
| desfase_pico_abs_mediana_h | 19 | 8 | 8 | 8 |
| NSE_1h | 0.998 | 0.957 | 0.953 | 0.983 |
| NSE_6h | 0.940 | 0.915 | 0.916 | 0.945 |
| NSE_12h | 0.821 | 0.839 | 0.846 | 0.864 |
| NSE_24h | 0.538 | 0.651 | 0.675 | 0.668 |
| NSE_48h | 0.150 | 0.407 | 0.412 | 0.400 |

## Error por régimen de caudal (test, %)

| régimen | Persistencia | DMEL (paper) | Ablación: asignación invertida | Ablación: sin MRS |
|---|---|---|---|---|
| bajo | 10.299 | 43.758 | 48.060 | 59.685 |
| medio | 16.011 | 29.961 | 29.502 | 28.514 |
| alto | 39.071 | 32.561 | 32.563 | 31.023 |

## Wilcoxon pareado por cuenca (test)

| comparación | mediana ΔNSE | cuencas donde gana | p-valor |
|---|---|---|---|
| DMEL (paper) vs Persistencia | 0.082 | 343/508 | 6.2e-19 |
| DMEL (paper) vs Ablación: asignación invertida | -0.00501 | 238/508 | 8.7e-02 |
| DMEL (paper) vs Ablación: sin MRS | -0.0094 | 209/508 | 2.6e-05 |

## Entrenamiento

| corrida | parámetros (M) | épocas entrenadas | mejor época | NSE* val (mejor) | NSE val mediana (mejor) | min por época | horas totales |
|---|---|---|---|---|---|---|---|
| DMEL (paper) | 3.677 | 16 | 11 | 0.399 | 0.635 | 2.015 | 0.537 |
| Ablación: asignación invertida | 3.675 | 10 | 5 | 0.390 | 0.637 | 1.915 | 0.319 |
| Ablación: sin MRS | 3.671 | 10 | 5 | 0.382 | 0.655 | 1.983 | 0.331 |

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
