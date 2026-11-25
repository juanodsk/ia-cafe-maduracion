# Dataset de madurez de frutos de café

Registro de la descarga Coffee Fruit Maturity v5 utilizada como punto de partida para comparar tres CNN y dos Transformers en la clasificación de madurez de frutos de café. Esta ficha conserva la configuración de Roboflow y los hallazgos de la revisión local del 24 de noviembre de 2026.

## Procedencia y formato

| Campo | Valor |
| --- | --- |
| Dataset | Coffee Fruit Maturity |
| Publicador en Roboflow | Ciencia Cafeto |
| Especie indicada | Coffea arabica |
| Versión descargada | v5, 17 de octubre de 2023 |
| Fecha de exportación indicada en el README | 18 de octubre de 2023, 02:59 GMT |
| Licencia indicada | CC BY 4.0 |
| Formato | COCO Segmentation |
| Carpeta local | `C:\Users\juanp\Desktop\juan-castro\imagenes` |
| Archivos de anotación | Un `_annotations.coco.json` en cada carpeta `train`, `valid` y `test` |

Fuente del dataset: [Coffee Fruit Maturity de Ciencia Cafeto](https://universe.roboflow.com/ciencia-cafeto/coffee-fruit-maturity-befkg). La versión y la configuración de esta descarga quedan respaldadas por los archivos locales y la captura conservada al final de esta ficha.

## Distribución de la descarga

| Conjunto | Carpeta | Imágenes | Porcentaje mostrado en Roboflow | Porcentaje calculado |
| --- | --- | ---: | ---: | ---: |
| Entrenamiento | `train` | 2.061 | 87 % | 87,33 % |
| Validación | `valid` | 201 | 9 % | 8,52 % |
| Prueba | `test` | 98 | 4 % | 4,15 % |
| Total | | 2.360 | 100 % | 100,00 % |

Los porcentajes mostrados son valores redondeados sobre los archivos exportados, que incluyen las variantes de entrenamiento. Las 2.360 imágenes no equivalen a 2.360 capturas independientes.

La partición 70 % para entrenamiento, 15 % para validación y 15 % para prueba ya está preparada en `dataset_preparado/`, con 1.651, 354 y 354 imágenes, respectivamente, tras excluir una copia exacta. La tabla anterior conserva el estado de la descarga original. La [documentación de la partición](particion_dataset.md) detalla el redondeo a imágenes enteras, la agrupación, los CSV y las comprobaciones.

## Preprocesamiento aplicado en Roboflow

| Operación | Configuración conservada |
| --- | --- |
| Autoorientación | Aplicada; el README también indica eliminación de la orientación EXIF |
| Redimensionamiento | Estiramiento a 640 × 640 píxeles, `Stretch to 640x640` |
| Modificación de clases | 0 clases reasignadas y 1 clase eliminada, `0 remapped, 1 dropped`, según la captura |

La captura no identifica el nombre de la clase eliminada. El registro de esa operación no permite deducir cuál era.

## Aumentos aplicados al entrenamiento

| Parámetro | Valor |
| --- | --- |
| Salidas por ejemplo de entrenamiento | 3 |
| Cizallamiento horizontal, `shear` | Entre −15° y +15° |
| Cizallamiento vertical, `shear` | Entre −15° y +15° |
| Ajuste aleatorio de exposición | Entre −12 % y +12 % |

Los nombres de archivo permiten identificar 687 nombres fuente en entrenamiento, cada uno con tres variantes: 687 × 3 = 2.061 archivos. Validación y prueba contienen 201 y 98 archivos, respectivamente, con una variante por nombre fuente dentro de cada conjunto.

## Clases y anotaciones disponibles

| ID COCO | Etiqueta | Interpretación en español | Entrenamiento | Validación | Prueba | Total de anotaciones |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `dry` | Seco | 1.295 | 109 | 45 | 1.449 |
| 2 | `overripe` | Sobremaduro | 755 | 35 | 25 | 815 |
| 3 | `ripe` | Maduro | 3.329 | 348 | 131 | 3.808 |
| 4 | `semi_ripe` | Semimaduro | 2.080 | 229 | 144 | 2.453 |
| 5 | `unripe` | Verde o inmaduro | 21.868 | 2.515 | 1.216 | 25.599 |
| | Total | | 29.327 | 3.236 | 1.561 | 34.124 |

Estos conteos corresponden a anotaciones de frutos e incluyen las variantes aumentadas de entrenamiento; no son conteos de frutos físicos únicos. Una imagen puede contener varios frutos. La categoría `fruit`, ID 0, es la categoría padre y tiene cero anotaciones. Esta descarga no contiene una clase anotada `diseased`.

La etiqueta `semi_ripe` es la candidata para representar el café pintón del proyecto; su correspondencia con el criterio de verde con rojo deberá comprobarse al revisar ejemplos de esa clase.

## Hallazgos para preparar las particiones

La revisión local encontró todas las imágenes referenciadas por los JSON, sin referencias inválidas de imagen o categoría, sin cajas de tamaño inválido y sin anotaciones que carecieran del campo de segmentación. Esta revisión estructural no constituye una validación agronómica de las etiquetas.

Se confirmó mediante SHA256 una imagen idéntica en entrenamiento y validación:

- Entrenamiento: `train/IMG_20230824_102533_jpg.rf.75f022757d3761e71846158374f53a29.jpg`.
- Validación: `valid/IMG_20230824_102533_jpg.rf.d8e60db633bbed18bb2c88dcb42ac6a7.jpg`.

El nombre fuente `IMG_20230824_102533_jpg` tiene tres variantes en entrenamiento y una imagen en validación. Esta coincidencia representa fuga de datos entre ambos conjuntos.

La partición preparada reúne las variantes y las capturas relacionadas identificadas por sus nombres y marcas de tiempo; conserva una sola copia de la imagen idéntica. Cada futuro recorte de fruto deberá heredar el conjunto de su imagen. El campo `date_captured` del JSON corresponde a la exportación y no debe utilizarse como fecha real de toma de las fotografías. Los detalles y las limitaciones de la agrupación están en la [documentación de la partición](particion_dataset.md).

## Fuentes locales conservadas

- [Captura de la distribución y configuración de Roboflow](assets/roboflow-v5-configuracion.png), aportada por el usuario.
- [Descripción del dataset](../imagenes/README.dataset.txt).
- [Configuración de exportación](../imagenes/README.roboflow.txt).
- [Anotaciones de entrenamiento](../imagenes/train/_annotations.coco.json).
- [Anotaciones de validación](../imagenes/valid/_annotations.coco.json).
- [Anotaciones de prueba](../imagenes/test/_annotations.coco.json).
