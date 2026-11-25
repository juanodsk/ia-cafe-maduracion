# Partición del dataset de madurez de café

La partición activa del proyecto está en `dataset_preparado/`. Se generó a partir de la descarga v5 conservada en `imagenes/`, con un objetivo de 70 % para entrenamiento, 15 % para validación y 15 % para prueba. La semilla es 42.

## Distribución preparada

| Conjunto | Carpeta | Imágenes | Porcentaje real | Grupos por captura | Anotaciones de frutos |
| --- | --- | ---: | ---: | ---: | ---: |
| Entrenamiento | `dataset_preparado/train` | 1.651 | 69,9873 % | 620 | 23.878 |
| Validación | `dataset_preparado/valid` | 354 | 15,0064 % | 125 | 5.116 |
| Prueba | `dataset_preparado/test` | 354 | 15,0064 % | 139 | 5.117 |
| Total | | 2.359 | 100 % | 884 | 34.111 |

Se excluyó una copia exacta de la descarga, que contenía 2.360 archivos. Con 2.359 imágenes no es posible representar exactamente 70 %, 15 % y 15 % mediante números enteros. Se aplicó el método de restos mayores: cada conjunto recibe primero la parte entera de su cuota y los archivos restantes se asignan a las cuotas con mayor fracción. El resultado es 1.651 / 354 / 354.

Los porcentajes corresponden a archivos de imagen únicos por contenido. Los grupos y los frutos tienen distribuciones propias porque una captura puede tener varias variantes y contener varios frutos. Las cinco clases están presentes en los tres conjuntos. El algoritmo busca aproximar también el reparto 70/15/15 de las anotaciones de cada clase, manteniendo intactos los grupos y los conteos de imágenes.

## Agrupación y duplicados

Las variantes con el mismo nombre anterior a `.rf.` permanecen juntas. Los nombres que comparten fecha y hora `AAAAMMDD_HHMMSS` también se agrupan, incluidos aliases como `1_20211204_100510` y `2_20211204_100510`. Las coincidencias exactas de SHA256 se unen antes de asignar los conjuntos.

La copia idéntica detectada entre entrenamiento y validación se conserva una sola vez en el dataset preparado. Se eligió la copia procedente de `valid`, con sus 13 anotaciones completas. Las dos copias tenían las mismas clases y cantidades, pero pequeñas diferencias de trazado; sus anotaciones no se sumaron. Los archivos y las dos versiones de las anotaciones originales permanecen en `imagenes/`.

La verificación exige que cada nombre fuente, marca de tiempo identificada, grupo y hash aparezca en un único conjunto. Los nombres no incluyen identificadores de finca, planta o fruto físico, por lo que estos controles no prueban independencia entre capturas distintas de una misma planta o sesión.

## Aumentos heredados de la versión descargada

La v5 ya contiene variantes de cizallamiento y exposición. La reorganización conserva esas imágenes, agrupadas con su captura. Los metadatos no permiten identificar de forma fiable cuál de las variantes es la imagen sin aumentos; por tanto, validación y prueba pueden contener aumentos heredados.

No se deben aplicar aumentos aleatorios adicionales a validación o prueba. Una evaluación que requiera exclusivamente imágenes sin aumentos necesita una nueva exportación sin aumentación. Esta limitación de la descarga queda separada del reparto 70/15/15, que ya está preparado.

## Archivos para reproducibilidad

| Archivo dentro de `dataset_preparado` | Contenido |
| --- | --- |
| `csv/train.csv`, `csv/valid.csv`, `csv/test.csv` | Asignación fija de imágenes a cada conjunto |
| `csv/imagenes.csv` | Manifiesto completo con rutas, hash, grupo, IDs originales y cantidades por clase |
| `csv/anotaciones.csv` | Una fila por fruto anotado, con clase, caja, imagen padre, grupo y procedencia |
| `csv/grupos.csv` | Grupos por captura, conjunto asignado, nombres fuente y conteos por clase |
| `csv/distribucion.csv` | Distribución de imágenes, grupos y anotaciones por conjunto |
| `csv/distribucion_clases.csv` | Cantidades de anotaciones y presencia de clases en imágenes |
| `csv/duplicados_excluidos.csv` | Copia conservada, copia excluida, hash y diferencias de anotación |
| `preparacion.json` | Parámetros, semilla, huellas de las anotaciones de origen, resumen y limitaciones |
| `train/_annotations.coco.json`, `valid/_annotations.coco.json`, `test/_annotations.coco.json` | COCO actualizado con IDs y referencias de las nuevas particiones |

Los CSV se pueden cargar con `pandas.read_csv`. Los conteos por clase describen objetos anotados; una imagen puede contener varias clases. La columna `images_containing_class` registra esa presencia y no debe sumarse como si cada imagen tuviera una sola clase.

## Reproducir y comprobar

Los scripts utilizan módulos incluidos en Node.js y se ejecutan desde la raíz del proyecto:

```powershell
node .\scripts\preparar_dataset.cjs
node .\scripts\verificar_dataset.cjs
```

La preparación rechaza una carpeta de salida existente para preservar la asignación guardada. Para reproducirla en otra carpeta:

```powershell
node .\scripts\preparar_dataset.cjs imagenes dataset_reproducido
node .\scripts\verificar_dataset.cjs dataset_reproducido
```

Con los mismos archivos de origen y parámetros se obtiene la misma asignación. El manifiesto `csv/imagenes.csv` generado tiene SHA256 `f99fcf50f70c45aad9eeffae1c857ad653af4bf64e017e1d5352220782ccb651`.

La comprobación contrasta los archivos con sus originales, verifica dimensiones JPEG, referencias COCO, conservación de etiquetas y geometrías, conteos de los CSV, ausencia de duplicados exactos y separación de grupos. Estos controles estructurales no evalúan la calidad agronómica de las etiquetas.

## Uso con CNN y Transformers

Los recortes de frutos y los generadores por lotes deben utilizar los conjuntos guardados. Cada recorte hereda el conjunto de su imagen padre; los frutos no se vuelven a repartir individualmente. Los cinco modelos utilizarán esta misma asignación para que su comparación sea consistente.

La optimización de hiperparámetros y cualquier validación cruzada deben mantener el conjunto de prueba reservado para la evaluación final. La preparación actual organiza los datos; no ejecuta entrenamiento ni selecciona hiperparámetros.
