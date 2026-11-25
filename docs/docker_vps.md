# Docker y despliegue en VPS

La imagen ejecuta React compilado y FastAPI en el mismo puerto, **8765**, con los cinco clasificadores y el ensemble. Está preparada para una VPS Linux con CPU; no necesita GPU ni CUDA. La imagen construida en este equipo es para Linux **amd64 / x86_64**. No se vuelve a entrenar ningún modelo.

## Desplegar directamente desde el repositorio

Con Git LFS, Docker y Docker Compose instalados en la VPS:

```bash
git lfs install
git clone https://github.com/juanodsk/ia-cafe-maduracion.git
cd ia-cafe-maduracion
git lfs pull
cp .env.example .env
docker compose up -d --build
```

Los archivos `.pt` y `.pth` están versionados mediante Git LFS. `git lfs pull` descarga los pesos reales, incluidos los tres backbones de `.cache/torch/hub/checkpoints/`. Sin ellos la comprobación SHA-256 del Dockerfile falla antes de producir una imagen incompleta. Los datasets completos no se requieren para ejecutar la app y no se publican en este repositorio. Consulta la [documentación de Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) para su funcionamiento.

## Paquete para transferir

`dist_deploy/cafe-madurez-vps.zip` contiene el Dockerfile, Compose, fuentes, ejemplo de validación y los **nueve archivos de pesos necesarios**. No incluye los datasets, entornos virtuales, datos de entrenamiento, resultados de experimentos ni `node_modules`.

El archivo `.cache/torch/...` incluido es necesario: ResNet50, DenseNet121 y ViT guardaron cabezas entrenadas por separado de sus backbones ImageNet. EfficientNet y Swin utilizan sus checkpoints completos de ajuste fino. `deploy/model_assets.json` conserva tamaño y SHA-256 de cada peso; la compilación comprueba que todos coincidan.

Para volver a generar el ZIP desde la carpeta original del proyecto:

```powershell
.\.venv\Scripts\python.exe deploy\empaquetar_vps.py --output dist_deploy\cafe-madurez-vps-nuevo.zip
```

## Subir y ejecutar

La VPS debe tener Docker Engine y el plugin Docker Compose instalados. Estas instrucciones asumen acceso a `docker` desde tu usuario. En tu equipo, transfiere el ZIP usando tus datos reales:

```powershell
scp .\dist_deploy\cafe-madurez-vps.zip usuario@IP_DE_LA_VPS:~/
```

En la VPS Linux:

```bash
unzip cafe-madurez-vps.zip
cd cafe-madurez
cp .env.example .env
docker compose up -d --build
docker compose ps
```

Abre **http://IP_DE_LA_VPS:8765**. El puerto TCP 8765 debe estar permitido en el firewall de la VPS y en el del proveedor. `.env` permite cambiar el puerto, la interfaz y los límites de CPU/memoria. Si eliges otro puerto, cambia también la URL.

La compilación descarga las imágenes base y las dependencias, por lo que necesita Internet. Los pesos ya vienen en el paquete: no se descargan ni se entrena al iniciar. La primera inferencia carga los clasificadores en memoria y tarda más. La app de la VPS indica que la foto se envía a ese servidor, donde se procesa; los resultados no se guardan automáticamente.

Compose configura un solo proceso de inferencia, reinicio automático y un límite de **2 CPU / 4 GB de RAM** para el contenedor. Estos son límites configurables, no una garantía de rendimiento para cualquier volumen de usuarios. Reserva memoria adicional para Linux y otros servicios. El ensemble y ViT consumen más recursos que EfficientNet B0. Mantén un solo worker; varios duplicarían los modelos en memoria.

## Usar una imagen ya construida

Si prefieres no compilar en la VPS, exporta la imagen probada en este equipo:

```powershell
docker image save -o dist_deploy\cafe-madurez-v1-amd64.tar cafe-madurez:v1
scp .\dist_deploy\cafe-madurez-v1-amd64.tar usuario@IP_DE_LA_VPS:~/
scp .\compose.yaml .\.env.example usuario@IP_DE_LA_VPS:~/
```

En la VPS x86_64:

```bash
docker image load -i cafe-madurez-v1-amd64.tar
mkdir -p cafe-madurez
mv compose.yaml .env.example cafe-madurez/
cd cafe-madurez
cp .env.example .env
docker compose up -d --no-build --pull never
```

La imagen incluye interfaz y pesos: no necesita montar carpetas del equipo original. Nunca montes todo `experimentos/` encima de la imagen, porque podrías ocultar sus pesos con una carpeta vacía.

## Dominio y HTTPS

Si ya utilizas un proxy Nginx en la VPS, establece `APP_BIND=127.0.0.1` en `.env`. Usa `deploy/nginx.conf.example` como referencia, reemplazando `cafe.example.com` por tu dominio. Configura el certificado HTTPS con la herramienta que utilices en tu servidor. La plantilla es solo una referencia HTTP para el proxy y no instala ni emite certificados.

El proxy admite archivos de hasta 16 MB y espera hasta 180 segundos por el análisis. La app limita las fotos a 15 MB y 20 megapíxeles. Con un proxy gestionado por otro contenedor, conecta ambos a su red Docker y apunta al puerto interno 8765 según la configuración de ese proxy.

## Comprobar y administrar

```bash
curl --fail http://127.0.0.1:8765/api/health
docker compose logs --tail=100 -f app
docker compose exec app python deploy/verify_models.py
docker compose restart app
docker compose down
```

En la interfaz, **Probar con una foto de ejemplo** comprueba el flujo completo de detección y clasificación. La foto pertenece al conjunto de validación; demostrar que se procesa no equivale a medir precisión en fotos nuevas.

La comprobación de estado indica que la API está disponible; los modelos se cargan de forma diferida al analizar una foto. Para una comprobación automática de inferencia desde el proyecto original:

```powershell
.\.venv\Scripts\python.exe scripts\verificar_app_fotos.py --url http://127.0.0.1:8765 --all-models
```

Referencias de implementación: [compilación Docker por etapas](https://docs.docker.com/build/building/multi-stage/), [Docker Compose](https://docs.docker.com/compose/intro/compose-application-model/) y [binarios PyTorch CPU](https://pytorch.org/get-started/previous-versions/).

## Verificación del paquete

Comprobación realizada el 25 de noviembre de 2026 en Docker Linux amd64:

- Compilación de React y arranque del contenedor como usuario `cafe`, sin privilegios de root.
- Siete pruebas de la API aprobadas dentro del contenedor, con red deshabilitada.
- Análisis HTTP correcto con los cinco modelos y con el ensemble usando únicamente CPU, con límite de 2 CPU y 4 GB de RAM. Todos localizaron los nueve frutos del ejemplo.
- Inferencia del ensemble completada con `--network none`: no necesita descargar pesos al ejecutar.
- Prueba en el navegador: aviso de procesamiento en servidor, foto de ejemplo, recuadros, etiquetas y porcentajes visibles; se mostró CPU como dispositivo.

El resultado trazable de la prueba HTTP está en `pruebas/docker_app/demo_resultado.json` del proyecto original. Los tiempos de esa ejecución fueron aproximadamente 4,2 s con EfficientNet y 12,7 s con el ensemble, incluyendo la carga al cambiar de modelo. Son referencias de este equipo con límites Docker; el rendimiento de otra VPS puede variar.
