FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    HF_HUB_CACHE=/models/hf/hub \
    PYTHONPATH=/app/runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY runtime/requirements-core.txt runtime/requirements.txt ./runtime/
RUN pip install --upgrade pip && pip install -r runtime/requirements.txt

COPY runtime/voice_service ./runtime/voice_service
COPY runtime/app.py ./runtime/app.py
COPY config ./config

RUN useradd --create-home --uid 1000 --gid voice voice && \
    mkdir -p /data/jobs /data/outputs /models/hf && \
    chown -R voice:voice /data /models /app

USER voice
EXPOSE 8889

CMD ["uvicorn", "voice_service.main:app", "--app-dir", "/app/runtime", "--host", "0.0.0.0", "--port", "8889"]
