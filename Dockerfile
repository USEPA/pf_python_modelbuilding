# Use Ubuntu's patched Python 3.12 runtime; keep compilers in the build stage.
FROM ubuntu:24.04 AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Refresh cached OS package layers for the security rebuild.
ARG SECURITY_REFRESH=2026-10-03
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates python3 libfreetype6 libfontconfig1 fonts-dejavu-core libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM runtime AS builder
# Refresh cached OS package layers for the security rebuild.
ARG SECURITY_REFRESH=2026-10-03
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip check && pip uninstall -y pip wheel

FROM runtime
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

COPY . .

EXPOSE 8080 9090

ENV MONITORING_PORT=9090

CMD ["sh", "-c", "export SERVICE_NAME=${SERVICE_NAME:-predictor_models}; export READINESS_URL=${READINESS_URL:-http://127.0.0.1:8080/api/predictor_models/version}; export PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus_multiproc}; mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\"; rm -f \"$PROMETHEUS_MULTIPROC_DIR\"/*.db; uvicorn --host ${MONITORING_HOST:-0.0.0.0} --port ${MONITORING_PORT:-9090} model_service_common.monitoring:monitoring_app & exec uvicorn --host 0.0.0.0 --port 8080 app:app"]
