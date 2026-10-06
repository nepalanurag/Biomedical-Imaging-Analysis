# Cloud runner image: the production pipeline plus the FastAPI service.
# Build (from the repo root):
#   docker build -f cloud/Dockerfile -t ctpipe-cloud .
# Run locally (needs CLOUD_STORAGE_BACKEND=local):
#   docker run --rm -p 8080:8080 \
#     -e CLOUD_STORAGE_BACKEND=local -e CLOUD_LOCAL_DIR=/tmp/cloud-local \
#     ctpipe-cloud
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libjpeg is needed by pylibjpeg-libjpeg at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

COPY pipeline/requirements.txt ./pipeline-requirements.txt
RUN pip install --no-cache-dir -r pipeline-requirements.txt

COPY cloud/requirements.txt ./cloud-requirements.txt
RUN pip install --no-cache-dir -r cloud-requirements.txt

COPY pipeline/ ./pipeline/
COPY cloud/ ./cloud/

# pipeline_data/ and job scratch are written at runtime; keep them out of the image.
VOLUME ["/app/pipeline_data"]

ENV PORT=8080
EXPOSE 8080

# One worker: jobs are CPU-bound and run up to ~15 minutes; Cloud Run
# scales instances instead. --timeout-keep-alive keeps long uploads alive.
CMD ["uvicorn", "cloud.app:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-keep-alive", "900"]
