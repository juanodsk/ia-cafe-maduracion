# Mejoras mediante fine tuning

Se ajustaron las últimas etapas de EfficientNet B0 y Swin T usando las particiones guardadas. No se utilizaron datos de prueba ni imágenes sintéticas. Las métricas corresponden a los mismos 5.116 frutos de validación de la primera ronda.

| Modelo | Accuracy antes | Accuracy después | F1 macro antes | F1 macro después | Precisión macro antes | Precisión macro después |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| efficientnet_b0 | 90.38 % | 91.65 % | 0.7280 | 0.7301 | 0.7207 | 0.7669 |
| swin_t | 90.73 % | 90.73 % | 0.7215 | 0.7215 | 0.7335 | 0.7335 |
| ensemble_cinco | 92.22 % | 92.42 % | 0.7510 | 0.7503 | 0.7706 | 0.7798 |

EfficientNet B0 conservó la segunda época de cuatro ejecutadas. Swin T detuvo el ajuste tras dos épocas sin mejora de F1 macro y conservó los pesos de referencia (época cero). El ensemble ganó aproximadamente 0,20 puntos porcentuales de accuracy y 0,92 puntos de precisión macro; F1 macro bajó 0,0008. Es una mejora pequeña en esta validación, sin evidencia todavía de una mejora sobre el conjunto de prueba.

## Resultados por clase del ensemble

| Clase | F1 antes | F1 después | Recall antes | Recall después |
| --- | ---: | ---: | ---: | ---: |
| dry | 0.8398 | 0.8370 | 0.7972 | 0.7926 |
| overripe | 0.4412 | 0.4271 | 0.3689 | 0.3361 |
| ripe | 0.8250 | 0.8342 | 0.8669 | 0.8897 |
| semi_ripe | 0.6693 | 0.6729 | 0.6929 | 0.6875 |
| unripe | 0.9799 | 0.9803 | 0.9771 | 0.9781 |

## Protocolo y límites

Se partió de los pesos preentrenados y las cabezas guardadas. Se mantuvieron congeladas las capas iniciales y las estadísticas de BatchNorm; las últimas etapas y la cabeza recibieron gradientes. AdamW utilizó learning rate de 0,00001 para el backbone y 0,00003 para la cabeza, batch 32, máximo seis épocas y parada tras dos épocas sin mejora de F1 macro. La normalización de características conservó las estadísticas de entrenamiento de la primera ronda.

La selección de checkpoint se hizo por F1 macro de validación; la accuracy y precisión macro se registran por separado. El ensemble sustituye únicamente las predicciones de los dos modelos afinados y mantiene las de los otros tres, con pesos uniformes.

La CV de la primera ronda no se presenta como CV de estos nuevos pesos. Esta comparación utiliza validación fija repetida; el 15 % de prueba sigue reservado para una evaluación final independiente. Se conservan las limitaciones de la descarga v5 y sus aumentos heredados.

[Comparación numérica](../experimentos/finetune_01/comparacion.csv) · [Gráfica](../experimentos/finetune_01/comparacion.png) · [Resultados y checkpoints](../experimentos/finetune_01/comparacion.json)
