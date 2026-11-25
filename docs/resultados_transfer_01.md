# Resultados de clasificación de madurez de café

El experimento `transfer_01` registra resultados de validación para 5 de 5 arquitecturas configuradas. Las métricas siguientes proceden de los artefactos guardados; los modelos pendientes se indican como tales.

## Alcance del entrenamiento

Se utilizan extractores preentrenados en ImageNet con sus pesos congelados. Se extraen características de cada fruto por lotes y se entrena una cabeza de clasificación para las cinco etapas de madurez. La normalización de características y los pesos de clase se calculan exclusivamente con las filas de entrenamiento de cada ajuste. Esta fase no actualiza todas las capas de las CNN o los Transformers.

Cada recorte hereda el conjunto y grupo de su imagen padre. La división fija es 70 % / 15 % / 15 % sobre imágenes exportadas únicas, con redondeo entero. El F1 macro de validación selecciona hiperparámetros, época y candidatos al ensemble; accuracy y balanced accuracy se presentan como medidas complementarias.

| Parámetro del experimento | Valor registrado |
| --- | --- |
| Semilla general | 42 |
| Dispositivo | NVIDIA GeForce RTX 4060 Ti |
| Ensayos solicitados por arquitectura | 12 |
| Máximo de épocas por ensayo | 12 |
| Folds solicitados | 5 |
| Fase del runner | `validation_complete` |

## Arquitecturas

| Arquitectura | Familia | Pesos de origen | Dimensión de características | Frutos de train / valid |
| --- | --- | --- | ---: | ---: |
| ResNet-50 | CNN | `ResNet50_Weights.IMAGENET1K_V2` | 2.048 | 23.878 / 5.116 |
| EfficientNet-B0 | CNN | `EfficientNet_B0_Weights.IMAGENET1K_V1` | 1.280 | 23.878 / 5.116 |
| DenseNet-121 | CNN | `DenseNet121_Weights.IMAGENET1K_V1` | 1.024 | 23.878 / 5.116 |
| ViT-B/16 | Transformer | `ViT_B_16_Weights.IMAGENET1K_V1` | 768 | 23.878 / 5.116 |
| Swin-T | Transformer | `Swin_T_Weights.IMAGENET1K_V1` | 768 | 23.878 / 5.116 |

Los tamaños de entrada, interpolación y normalización corresponden al preset oficial de los pesos seleccionado para cada arquitectura. La extracción de características utiliza ese preprocesamiento determinista, sin aumentos aleatorios nuevos.

## Resultados de validación

| Arquitectura | Accuracy | Balanced accuracy | F1 macro | F1 ponderado | F1 macro CV media ± DE | Época elegida |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientNet-B0 | 90,38 % | 74,33 % | 0,7280 | 0,9076 | 0,7311 ± 0,0247 | 3 |
| DenseNet-121 | 90,85 % | 72,18 % | 0,7227 | 0,9099 | 0,7435 ± 0,0260 | 7 |
| Swin-T | 90,73 % | 71,56 % | 0,7215 | 0,9078 | 0,7434 ± 0,0300 | 9 |
| ViT-B/16 | 90,77 % | 71,58 % | 0,7182 | 0,9083 | 0,7316 ± 0,0294 | 11 |
| ResNet-50 | 90,15 % | 69,04 % | 0,7006 | 0,9008 | 0,7108 ± 0,0228 | 8 |

![Comparación de F1 macro en validación y CV](C:/Users/juanp/Desktop/juan-castro/experimentos/transfer_01/figuras/comparacion_validacion.png)

La validación cruzada utiliza `StratifiedGroupKFold` dentro de train: cada grupo completo se mantiene en un solo fold. Los hiperparámetros y el presupuesto de épocas se fijan con la validación separada y se reutilizan en los folds. Por ello, la media y la desviación estándar describen una CV condicionada a esa selección, no una búsqueda anidada. La desviación estándar es muestral entre folds; no es un intervalo de confianza ni variación entre cinco semillas completas del backbone.

## Hiperparámetros seleccionados

Hyperband interrumpe ensayos menos prometedores y prioriza recursos por épocas. En esta fase los hiperparámetros optimizados pertenecen a la cabeza de clasificación: tasa de aprendizaje, tamaño de lote, weight decay, dropout, dimensión oculta y ponderación de clases.

| Arquitectura | Learning rate | Batch | Weight decay | Dropout | Dimensión oculta | Potencia de peso de clase | Ensayos completos / podados |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 | 0.000846801 | 128 | 0.000248104 | 0,00 | 256 | 1,00 | 4 / 8 |
| EfficientNet-B0 | 0.000112458 | 128 | 7.06897e-06 | 0,20 | 256 | 0,50 | 3 / 9 |
| DenseNet-121 | 0.000846801 | 128 | 0.000248104 | 0,00 | 256 | 1,00 | 3 / 9 |
| ViT-B/16 | 0.000200587 | 128 | 0.00433792 | 0,20 | 256 | 0,50 | 4 / 8 |
| Swin-T | 0.000112458 | 128 | 7.06897e-06 | 0,20 | 256 | 0,50 | 2 / 10 |

## Curvas de aprendizaje

![Pérdida de entrenamiento y F1 macro de validación por arquitectura](C:/Users/juanp/Desktop/juan-castro/experimentos/transfer_01/figuras/curvas_aprendizaje.png)

Cada panel muestra el historial real del ajuste que reproduce el ensayo seleccionado. La línea vertical marca la época del checkpoint elegido. La pérdida de train y el F1 macro de validación utilizan ejes y unidades distintos.

## Ensembles y selección

El ensemble heterogéneo de cinco arquitecturas promedia sus probabilidades con igual peso por modelo. Los promedios de familia CNN y Transformer combinan arquitecturas distintas dentro de esa familia; pertenecer a la misma familia no los convierte en ensembles homogéneos.

El híbrido de familias asigna la mitad del peso total a las CNN y la otra mitad a los Transformers. Es un promedio jerárquico de probabilidades, equivalente a pesos 1/6 para cada CNN y 1/4 para cada Transformer; no es una nueva red de fusión entrenada.

El ensemble homogéneo promedia las cabezas de la mejor arquitectura individual, elegida por validación, entrenadas en los folds. Estas cabezas comparten el extractor ImageNet congelado y difieren en sus subconjuntos de entrenamiento.

| Ensemble | Componentes | F1 macro valid | Accuracy valid |
| --- | --- | ---: | ---: |
| Heterogéneo de cinco arquitecturas | ResNet-50, EfficientNet-B0, DenseNet-121, ViT-B/16, Swin-T | 0,7510 | 92,22 % |
| Promedio de las tres CNN | ResNet-50, EfficientNet-B0, DenseNet-121 | 0,7503 | 91,81 % |
| Promedio de los dos Transformers | ViT-B/16, Swin-T | 0,7311 | 91,58 % |
| Híbrido con igual peso por familia | ResNet-50, EfficientNet-B0, DenseNet-121, ViT-B/16, Swin-T | 0,7450 | 92,10 % |
| Homogéneo de clasificadores por fold | EfficientNet-B0; 5 cabezas por fold | 0,7225 | 90,32 % |

Candidato seleccionado por validación: **Heterogéneo de cinco arquitecturas**, con F1 macro 0,7510. Esta selección precede a la apertura de test y no se basa en sus resultados.

## Conjunto de prueba

El conjunto de prueba permanece reservado para una evaluación final después de fijar modelos, hiperparámetros y ensembles. Las tablas y figuras anteriores no utilizan predicciones ni métricas de test.

## Artefactos de validación y reproducibilidad

- **ResNet-50**: [resultado](<../experimentos/transfer_01/resnet50/resultado.json>), [matriz de confusión](<../experimentos/transfer_01/resnet50/validacion/confusion_matrix.png>), [predicciones de validación](<../experimentos/transfer_01/resnet50/validacion/predictions.csv>), [métricas por fold](<../experimentos/transfer_01/resnet50/cross_validation/fold_metrics.csv>).
- **EfficientNet-B0**: [resultado](<../experimentos/transfer_01/efficientnet_b0/resultado.json>), [matriz de confusión](<../experimentos/transfer_01/efficientnet_b0/validacion/confusion_matrix.png>), [predicciones de validación](<../experimentos/transfer_01/efficientnet_b0/validacion/predictions.csv>), [métricas por fold](<../experimentos/transfer_01/efficientnet_b0/cross_validation/fold_metrics.csv>).
- **DenseNet-121**: [resultado](<../experimentos/transfer_01/densenet121/resultado.json>), [matriz de confusión](<../experimentos/transfer_01/densenet121/validacion/confusion_matrix.png>), [predicciones de validación](<../experimentos/transfer_01/densenet121/validacion/predictions.csv>), [métricas por fold](<../experimentos/transfer_01/densenet121/cross_validation/fold_metrics.csv>).
- **ViT-B/16**: [resultado](<../experimentos/transfer_01/vit_b_16/resultado.json>), [matriz de confusión](<../experimentos/transfer_01/vit_b_16/validacion/confusion_matrix.png>), [predicciones de validación](<../experimentos/transfer_01/vit_b_16/validacion/predictions.csv>), [métricas por fold](<../experimentos/transfer_01/vit_b_16/cross_validation/fold_metrics.csv>).
- **Swin-T**: [resultado](<../experimentos/transfer_01/swin_t/resultado.json>), [matriz de confusión](<../experimentos/transfer_01/swin_t/validacion/confusion_matrix.png>), [predicciones de validación](<../experimentos/transfer_01/swin_t/validacion/predictions.csv>), [métricas por fold](<../experimentos/transfer_01/swin_t/cross_validation/fold_metrics.csv>).
- **Heterogéneo de cinco arquitecturas**: [matriz de confusión de validación](<../experimentos/transfer_01/ensembles/heterogeneo_cinco/validacion/confusion_matrix.png>).
- **Promedio de las tres CNN**: [matriz de confusión de validación](<../experimentos/transfer_01/ensembles/familia_cnn/validacion/confusion_matrix.png>).
- **Promedio de los dos Transformers**: [matriz de confusión de validación](<../experimentos/transfer_01/ensembles/familia_transformer/validacion/confusion_matrix.png>).
- **Híbrido con igual peso por familia**: [matriz de confusión de validación](<../experimentos/transfer_01/ensembles/hibrido_familias/validacion/confusion_matrix.png>).
- **Homogéneo de clasificadores por fold**: [matriz de confusión de validación](<../experimentos/transfer_01/ensembles/homogeneo_cinco_folds/validacion/confusion_matrix.png>).

SHA256 del manifiesto de imágenes: `f99fcf50f70c45aad9eeffae1c857ad653af4bf64e017e1d5352220782ccb651`.

SHA256 del manifiesto de anotaciones: `38392b94c64150c39138d2a142c06ca69c60fc8337a780fe56ba8a1d5269b2b9`.

## Límites y trabajo pendiente

La descarga v5 contiene aumentos previos de cizallamiento y exposición; conservar los grupos evita compartir las variantes identificadas entre conjuntos, pero no recupera originales sin aumentos. Tampoco hay identificadores de finca, planta o fruto físico que permitan garantizar independencia entre todas las capturas. El desbalance de clases se aborda con pesos calculados en train y se hace visible mediante F1 macro y las matrices de confusión.

Estos resultados, cuando estén completos, corresponden a aprendizaje por transferencia con extractores congelados y cabezas aprendidas. El ajuste de todas las capas del backbone sería una fase experimental distinta y debe registrarse por separado.

Las extensiones siguientes utilizan únicamente entrenamiento y validación. Sus presupuestos son pilotos distintos; no constituyen una comparación con igual cómputo frente al estudio centralizado.

## Experimentos GAN federado y distribuido

| Experimento | Estado | Macro F1 de validación |
| --- | --- | ---: |
| gan | complete | No aplica |
| gan_ablation | complete | 0.7163 |
| federated | complete | 0.3050 |
| distributed | complete | 0.7029 |

La GAN generó 1024 imágenes de las cuatro clases minoritarias, restringidas a train. El piloto no garantiza calidad agronómica de las etiquetas sintéticas.

**Revisión visual de la GAN:** Las muestras presentan ruido y estructuras poco definidas; no permiten validar visualmente los estados de madurez solicitados. Solo ablacion experimental; no recomendar como aumento validado ni sustituir datos reales.

La ablación con sintéticos cambió macro F1 en -0.0116 frente al mismo modelo base, con la misma semilla, parámetros y presupuesto de épocas. No se incorpora la GAN global a los folds de CV.

FedAvg simula tres clientes con grupos separados en un equipo. DDP utiliza dos procesos CPU reales con Gloo en ese mismo equipo; no es un despliegue en varias fincas ni una prueba de escalabilidad entre máquinas.

[Resumen trazable de extensiones](<../experimentos/transfer_01/resumen_extensiones.json>).

## Referencias técnicas

- [ResNet-50 en TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
- [EfficientNet-B0 en TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).
- [DenseNet-121 en TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html).
- [ViT-B/16 en TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html).
- [Swin-T en TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_t.html).
- [HyperbandPruner de Optuna](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html).
- [StratifiedGroupKFold de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
