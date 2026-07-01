FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libfontconfig1 \
    fonts-dejavu-core \
    libstdc++6 \
    libgomp1 \
    && apt-get remove --allow-remove-essential -y perl-base libsqlite3-0 ncurses-base ncurses-bin libncursesw6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8080 9090

ENV MONITORING_PORT=9090

CMD ["sh", "-c", "export SERVICE_NAME=${SERVICE_NAME:-predictor_models}; export READINESS_URL=${READINESS_URL:-http://127.0.0.1:8080/api/predictor_models/version}; export PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus_multiproc}; mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\"; rm -f \"$PROMETHEUS_MULTIPROC_DIR\"/*.db; uvicorn --host ${MONITORING_HOST:-0.0.0.0} --port ${MONITORING_PORT:-9090} model_service_common.monitoring:monitoring_app & exec uvicorn --host 0.0.0.0 --port 8080 app:app"]
