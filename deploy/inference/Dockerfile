FROM python:3.12-slim@sha256:6b25cbe6577518889d553cdd5a5da97d259348d8f6e45b527dd731caaba27a07
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/inference/requirements-cpu.lock /app/requirements-cpu.lock
COPY wheelhouse/ /wheelhouse/
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheelhouse --require-hashes -r /app/requirements-cpu.lock && python -m pip check
COPY src/ /app/src/
ENV PYTHONPATH=/app/src HOME=/tmp YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib YOLO_OFFLINE=True
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "cloud.server:from_environment", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8", "--timeout-keep-alive", "2"]
