FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

ADD sdk/contest-harness-sdk-1.0.4.tar.gz /tmp/
RUN python3 -m pip install --no-cache-dir --only-binary=:all: numpy==2.5.5 \
    && python3 -m pip install --no-cache-dir /tmp/contest-harness-sdk-1.0.4 \
    && rm -rf /tmp/contest-harness-sdk-1.0.4 \
    && mkdir -p /workspace \
    && chown 65534:65534 /workspace

WORKDIR /app
COPY app/harness.py app/contest_submission.py /app/
COPY app/a /app/a
COPY app/b /app/b
COPY app/c /app/c

USER 65534
ENTRYPOINT ["python3", "/app/harness.py"]
