# Production image for the DICOM ingestion pipeline.
# Build:  docker build -t ctpipe .
# Test:   docker run --rm ctpipe pytest tests/ -q
# Run:    docker run --rm -v /data/dicom:/data/dicom:ro -v $PWD/pipeline_data:/app/pipeline_data \
#           ctpipe python -m pipeline.ingest --dicom-root /data/dicom
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libjpeg is needed by pylibjpeg-libjpeg at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

COPY pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./pipeline/
COPY tests/ ./tests/
COPY dvc.yaml params.yaml ./

# pipeline_data/ is written at runtime; keep it out of the image.
VOLUME ["/app/pipeline_data"]

CMD ["python", "-m", "pipeline.ingest", "--help"]
