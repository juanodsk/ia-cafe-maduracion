# Metodología de entrenamiento de madurez de café

Los experimentos comparan tres CNN y dos Transformers para clasificar cada fruto como seco, sobremaduro, maduro, semimaduro o verde. Se utilizan las imágenes y los grupos de la partición 70/15/15 ya guardada. Cada fruto hereda el conjunto de su imagen; los recortes no se reparten de nuevo.

## Preparación y carga por lotes

El conjunto de entrenamiento contiene 23.878 frutos anotados y el de validación 5.116. El generador abre la fotografía correspondiente, limita la caja al borde de la imagen y recorta el fruto bajo demanda. Una caché pequeña por proceso evita conservar todas las imágenes en memoria. El preprocesamiento oficial de cada modelo produce entradas RGB de 224 × 224 píxeles.

Se conservan las cinco clases y todos los recortes de entrenamiento y validación. Hay 531 cajas de entrenamiento y 111 de validación que exceden ligeramente el borde; se limitan al área disponible. Treinta recortes de entrenamiento tienen un lado menor de ocho píxeles, lo que puede dificultar su clasificación. La auditoría se guarda en `experimentos/auditoria_recortes.json`.

La descarga v5 incluye aumentos previos de cizallamiento y exposición. Las variantes permanecen en un mismo conjunto, y esta primera ronda no añade alteraciones de color ni aumentos aleatorios de evaluación. La agrupación evita cruces identificables por nombre, marca de tiempo y hash; no demuestra independencia entre plantas o frutos físicos que carecen de identificadores.

## Tres CNN y dos Transformers

| Familia | Arquitectura | Pesos de origen fijados | Dimensión de características |
| --- | --- | --- | ---: |
| CNN | ResNet50 | ImageNet 1K V2 | 2.048 |
| CNN | EfficientNet B0 | ImageNet 1K V1 | 1.280 |
| CNN | DenseNet121 | ImageNet 1K V1 | 1.024 |
| Transformer | ViT B 16 | ImageNet 1K V1 | 768 |
| Transformer | Swin T | ImageNet 1K V1 | 768 |

La primera ronda utiliza transfer learning con el extractor preentrenado congelado. Se aprende una cabeza de clasificación específica para el café. Esta estrategia no actualiza todos los pesos de las redes. Permite comparar arquitecturas y repetir los ajustes de la cabeza sin recalcular las imágenes en cada ensayo.

La extracción utiliza lotes en la GPU y guarda matrices numéricas en disco con hashes y metadatos de procedencia. Las estadísticas de normalización de esas características se calculan exclusivamente con los datos de entrenamiento de cada ajuste. BatchNorm, dropout y stochastic depth del extractor permanecen en modo de evaluación.

## Hyperband y selección de hiperparámetros

Cada arquitectura tiene un estudio de 12 configuraciones, hasta 12 épocas por configuración. Optuna utiliza muestreo aleatorio con semilla 42 y Hyperband, con recurso mínimo de dos épocas y factor de reducción tres. Algunas configuraciones terminan antes mediante poda.

| Hiperparámetro de la cabeza | Espacio de búsqueda |
| --- | --- |
| Learning rate de AdamW | De 0,0001 a 0,03, escala logarítmica |
| Batch size | 128 o 256 |
| Weight decay | De 0,000001 a 0,01, escala logarítmica |
| Dropout | 0 o 0,2 |
| Capa oculta | Sin capa oculta o 256 unidades con GELU |
| Exponente de ponderación de clases | 0,5 o 1,0 |
| Épocas | Hasta 12; se conserva la época con mejor macro F1 de validación |

Los pesos de clase se calculan con el subconjunto de entrenamiento. La selección utiliza macro F1 de validación, porque el café verde constituye la mayoría de las anotaciones. También se registran accuracy, balanced accuracy, weighted F1 y resultados por clase. No se selecciona un modelo por su resultado en prueba.

Los estudios guardan su base SQLite, ensayos completos y podados, parámetros, semillas e historiales CSV. La cabeza ganadora se vuelve a entrenar con la misma semilla para comprobar que se reproduce antes de guardar su checkpoint.

## Validación cruzada

Además del holdout 70/15/15, se ejecuta StratifiedGroupKFold de cinco folds únicamente dentro del 70 % de entrenamiento. Los grupos de captura se mantienen íntegros en cada fold; la estratificación intenta conservar la distribución de clases sin dividir los grupos.

Cada fold ajusta su normalización y pesos de clase con su propio subconjunto de entrenamiento. La configuración y el número de épocas permanecen fijos, tomados de la selección con la validación separada. Se guardan asignaciones por fruto y grupo, predicciones fuera de fold, métricas, media y desviación estándar. Esta es una CV condicionada a la configuración seleccionada, no una validación cruzada anidada de toda la búsqueda.

Las otras técnicas de la guía tienen usos distintos:

| Técnica | Relación con este proyecto |
| --- | --- |
| Holdout | Corresponde a las particiones fijas de entrenamiento, validación y prueba |
| K fold | Se implementa su variante estratificada por grupos con cinco folds |
| Stratified k fold | Se añade la restricción de grupos para reunir variantes y capturas relacionadas |
| Leave one out y leave p out | No se ejecutan en esta ronda; requerirían muchos ajustes y deben respetar grupos |
| Monte Carlo o shuffle split | No se ejecuta; repite divisiones aleatorias, que pueden mantener una proporción fija |
| Time series o rolling | No se ejecuta; este objetivo es clasificación visual y las fechas reales no están garantizadas por los metadatos de exportación |

## Ensembles

El soft voting promedia probabilidades en el orden fijo de las cinco clases y exige que las filas correspondan al mismo fruto. Se comparan el promedio uniforme de las cinco arquitecturas, los promedios de la familia CNN y de la familia Transformer, y una combinación jerárquica que asigna 50 % del peso a cada familia.

El ensemble homogéneo promedia las cinco cabezas de los folds de una misma arquitectura, elegida con validación. Los promedios de varias arquitecturas de la familia CNN o Transformer se denominan ensembles de familia; compartir familia no significa que sus arquitecturas sean idénticas. El término híbrido se utiliza aquí para la combinación ponderada de ambas familias.

## Experimento con GAN

La GAN condicional genera imágenes de 64 × 64 píxeles a partir de ruido y una etiqueta de madurez. Generador y discriminador se entrenan exclusivamente con frutos del conjunto de entrenamiento. El piloto previsto utiliza cinco épocas de 80 pasos, con lotes de 64 y muestreo balanceado de las clases reales.

Los sintéticos de las clases minoritarias forman un conjunto adicional de entrenamiento. La ablación compara una cabeza entrenada con datos reales frente a otra con reales y sintéticos, manteniendo arquitectura, parámetros, semilla y presupuesto de épocas del modelo base. La comparación se realiza sobre la validación real fija.

Una GAN entrenada con todo el conjunto de entrenamiento no puede aportar sintéticos a los folds de CV de ese mismo conjunto: habría visto datos de los folds retenidos. La ablación con GAN no se presenta como CV. Condicionar una imagen por clase tampoco garantiza por sí solo su madurez agronómica; la cuadrícula generada y los resultados de la ablación deben acompañar la interpretación.

## Experimentos federado y distribuido

El experimento federado simula tres clientes con grupos de captura separados. Cada cliente ajusta una copia de la cabeza y FedAvg agrega sus parámetros según el número de muestras. Se guardan la distribución de los clientes y las métricas de validación por ronda. Es una simulación local; no representa fincas reales, redes geográficas ni garantías de privacidad diferencial.

El experimento distribuido utiliza dos procesos CPU con PyTorch DistributedDataParallel y Gloo en el mismo equipo. Los procesos distribuyen las muestras y sincronizan gradientes durante el entrenamiento. Esto permite demostrar ejecución distribuida sin suponer la existencia de varias GPU. Los presupuestos de ambos pilotos se registran y sus métricas no se interpretan como una comparación de eficiencia con igual cómputo frente a los estudios centralizados.

## Ejecución y artefactos

El entorno local utiliza Python 3.12, PyTorch 2.9.1 con CUDA 12.8 y TorchVision 0.24.1. Las dependencias directas están en `requirements.txt` y el entorno instalado se registra en `requirements.lock.txt`.

Para el piloto distribuido en Windows se creó `.venv_ddp/`, separado del entorno CUDA, con PyTorch 2.7.1 CPU y TorchVision 0.22.1 CPU. La instalación CUDA 2.9.1 devolvió `makeDeviceForHostname(): unsupported gloo device` incluso sin el aislamiento de red. `requirements-ddp.txt` conserva las dependencias del entorno alternativo. El lanzador de extensiones selecciona ese intérprete únicamente para la fase distribuida y registra su versión en el resultado.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -u -m cafe_ml.experiment download
.\.venv\Scripts\python.exe -u -m cafe_ml.experiment train
```

El directorio de esta ronda es `experimentos/transfer_01`. El archivo `estado.json` indica su etapa. Los resultados numéricos pertenecen al informe generado desde los artefactos del experimento; este protocolo no presupone que las ejecuciones previstas hayan terminado.

El conjunto de prueba se abre mediante una fase explícita después de cerrar la selección. No se utiliza para ajustar parámetros, sintetizar datos, formar clientes o entrenar los modelos.

## Referencias

- [Transfer learning para visión por computadora en PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- [Hyperband en Optuna](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html).
- [StratifiedGroupKFold en scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784), Mirza y Osindero.
- [FedAvg y aprendizaje con datos descentralizados](https://proceedings.mlr.press/v54/mcmahan17a.html), McMahan y colaboradores.
- [DistributedDataParallel de PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
