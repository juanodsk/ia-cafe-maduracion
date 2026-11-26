# Clasificación de madurez de frutos de café

Proyecto para comparar tres CNN y dos Transformers usando imágenes de frutos de café en cinco etapas de madurez.

## Ejecutar desde GitHub con Docker

Instala Git, **Git LFS**, Docker y Docker Compose. Clona con los pesos grandes y arranca la aplicación en una VPS Linux x86_64:

```bash
git lfs install
git clone https://github.com/juanodsk/ia-cafe-maduracion.git
cd ia-cafe-maduracion
git lfs pull
cp .env.example .env
docker compose up -d --build
```

Abre `http://IP_DE_LA_VPS:8765`, permitiendo el puerto en el firewall. La imagen usa CPU y contiene los cinco modelos más el ensemble. Los modelos se distribuyen mediante Git LFS; descargar el ZIP de código de GitHub puede entregar punteros en lugar de los pesos. La [guía de Docker y VPS](docs/docker_vps.md) incluye configuración, límites de recursos, dominio y comprobaciones.

El repositorio conserva código, anotaciones COCO, manifiestos y distribuciones CSV, resultados y checkpoints. Las fotos del dataset, embeddings regenerables, bases locales de Hyperband, entornos virtuales, secretos y paquetes de despliegue generados permanecen fuera de Git. La aplicación de fotos funciona con los pesos y el ejemplo incluidos; para repetir entrenamientos debes restaurar las imágenes del dataset documentado en `imagenes/` y `dataset_preparado/`.

## Cámara en vivo

La [guía de cámara](docs/camara.md) explica la detección automática y los recuadros por madurez. Desde consola:

```powershell
.\.venv\Scripts\python.exe -m cafe_ml.camera --fuente 0
```

`Q` cierra y `S` guarda una captura. La cámara USB externa puede tener índice 1. El modo automático utiliza el detector de frutos adicional y el clasificador afinado; el modo `--manual` clasifica solo la zona fija de la pantalla.

## Dataset activo

Utilizar `dataset_preparado/`, que contiene la partición 70 % / 15 % / 15 %: **1.651 imágenes de entrenamiento, 354 de validación y 354 de prueba**. Cada conjunto incluye sus imágenes y anotaciones COCO Segmentation.

La descarga original está conservada en `imagenes/`. El dataset preparado excluye una copia exacta y mantiene juntas las variantes y las capturas relacionadas identificadas por sus nombres y marcas de tiempo.

## Documentación

- [Informe técnico completo y guion de 20 diapositivas en Word](docs/documentacion_tecnica_cafe.docx).

- [Partición preparada y reproducibilidad](docs/particion_dataset.md).
- [Datos de la descarga y captura original de Roboflow](docs/dataset.md).
- [Distribución en CSV](dataset_preparado/csv/distribucion.csv).
- [Distribución por clase en CSV](dataset_preparado/csv/distribucion_clases.csv).
- [Metodología de entrenamiento y experimentos](docs/metodologia_entrenamiento.md).

La versión v5 descargada ya incluía imágenes aumentadas. La documentación de la partición explica cómo se conservan y qué implica para la evaluación.

## Comprobar el dataset

```powershell
node .\scripts\verificar_dataset.cjs
```

La semilla y los manifiestos se conservan para que los cinco modelos utilicen exactamente la misma división.

## Entrenamiento de los cinco modelos

Las arquitecturas son ResNet50, EfficientNet B0 y DenseNet121, más ViT B 16 y Swin T. La primera ronda conserva los extractores ImageNet congelados y aprende cabezas de clasificación del café. Compara hiperparámetros con Hyperband y ejecuta cinco folds por grupos dentro de entrenamiento.

El entorno aislado está en `.venv/`; sus versiones se registran en `requirements.lock.txt`. Para reproducir la ronda en una carpeta nueva:

```powershell
.\.venv\Scripts\python.exe -u -m cafe_ml.experiment train --output experimentos\transfer_reproducido
```

El experimento predeterminado está en `experimentos/transfer_01/`. Incluye checkpoints, historiales, estudios Hyperband, asignaciones de folds, métricas y predicciones trazables. El archivo `estado.json` indica la etapa, y `comparacion_validacion.csv` registra los modelos terminados. La prueba queda reservada hasta una fase final explícita.

## Pruebas del código

Para subir fotos desde una interfaz React local, ejecuta `iniciar_app.cmd` y abre **http://127.0.0.1:8765**. Verás recuadros por fruto, madurez, puntuaciones y opciones para descargar la foto anotada o el informe JSON. Consulta la [guía de la aplicación](docs/app_fotos.md).

Para ejecutarla con Docker o subirla a una VPS Linux, consulta la [guía de despliegue](docs/docker_vps.md). `docker compose up -d --build` compila React e incluye los pesos existentes en una imagen de inferencia para CPU.

Para probar una foto propia, consulta la [guía de predicción](pruebas/README.md):

```powershell
.\.venv\Scripts\python.exe -m cafe_ml.predict --imagen "pruebas\mi_cafe.jpg" --modelo ensemble
```

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location).Path '.cache\matplotlib'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Los módulos de GAN, simulación federada y ejecución distribuida complementan la comparación centralizada. Su estado y presupuesto se registran en los artefactos correspondientes; el informe de resultados distingue ejecuciones terminadas de pendientes.

```powershell
.\.venv\Scripts\python.exe -u -m cafe_ml.extensions --phase all
.\.venv\Scripts\python.exe scripts\generar_informe.py
```

Consultar el [informe de resultados de esta ronda](docs/resultados_transfer_01.md), generado con métricas reales. `resumen_extensiones.json` registra por separado la GAN, su ablación, FedAvg y DDP. Ninguno de esos pasos utiliza el conjunto de prueba.

## Ajuste de las últimas capas

La ronda `experimentos/finetune_01/` afinó las últimas etapas de EfficientNet B0 y Swin T. EfficientNet mejoró de 90,38 % a 91,65 % de accuracy; Swin T conservó su checkpoint anterior porque el ajuste no mejoró F1 macro. El ensemble actualizado alcanzó 92,42 % de accuracy frente al 92,22 % anterior, con F1 macro prácticamente igual y ligeramente menor.

Ver la [comparación detallada antes y después](docs/resultados_finetune_01.md). Los checkpoints completos se pueden restaurar con `cafe_ml.finetune.load_finetuned`. El 15 % de prueba sigue reservado.

```powershell
.\.venv\Scripts\python.exe -u -m cafe_ml.finetune --output experimentos\finetune_reproducido
.\.venv\Scripts\python.exe scripts\verificar_finetune.py
```
