# syntax=docker/dockerfile:1
FROM node:24-bookworm-slim AS frontend
WORKDIR /web
RUN npm install --global pnpm@11.25.0
COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/index.html web/vite.config.js ./
COPY web/src ./src
RUN pnpm build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/app/.cache/torch \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_AUTOINSTALL=false \
    CAFE_DEPLOYMENT=server \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 cafe \
    && useradd --uid 10001 --gid cafe --create-home cafe
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir torch==2.9.1+cpu torchvision==0.24.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && pip check

COPY cafe_ml/*.py ./cafe_ml/
COPY --from=frontend /web/dist ./web/dist
COPY experimentos/detector_01/weights/best.pt ./experimentos/detector_01/weights/best.pt
COPY experimentos/finetune_01/efficientnet_b0/checkpoint.pt ./experimentos/finetune_01/efficientnet_b0/checkpoint.pt
COPY experimentos/finetune_01/swin_t/checkpoint.pt ./experimentos/finetune_01/swin_t/checkpoint.pt
COPY experimentos/transfer_01/resnet50/checkpoint.pt ./experimentos/transfer_01/resnet50/checkpoint.pt
COPY experimentos/transfer_01/densenet121/checkpoint.pt ./experimentos/transfer_01/densenet121/checkpoint.pt
COPY experimentos/transfer_01/vit_b_16/checkpoint.pt ./experimentos/transfer_01/vit_b_16/checkpoint.pt
COPY .cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth ./.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
COPY .cache/torch/hub/checkpoints/densenet121-a639ec97.pth ./.cache/torch/hub/checkpoints/densenet121-a639ec97.pth
COPY .cache/torch/hub/checkpoints/vit_b_16-c867db91.pth ./.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth
COPY pruebas/camara/demo_validacion.jpg ./pruebas/camara/demo_validacion.jpg
COPY deploy/verify_models.py deploy/model_assets.json ./deploy/
USER cafe:cafe
RUN python deploy/verify_models.py \
    && python -c "from cafe_ml.web_api import app; assert app is not None"

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=4).close()"]
# One worker keeps one model cache and one inference lock per container.
CMD ["python", "-m", "uvicorn", "cafe_ml.web_api:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1", "--limit-concurrency", "8", "--timeout-keep-alive", "5"]
