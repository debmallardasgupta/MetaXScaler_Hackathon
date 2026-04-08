# ── Email Triage OpenEnv Docker Image ────────────────────────────────────────
#
# Build:  docker build -t email-triage-env .
# Run:    docker run -p 7860:7860 email-triage-env
#
# Environment variables (set at runtime):
#   MY_ENV_V4_TASK    Active task: email_urgent | email_route | email_triage
#                     (default: email_urgent)
#   HF_TOKEN          Hugging Face / API key for inference.py
#   API_BASE_URL      LLM endpoint (default: https://router.huggingface.co/v1)
#   MODEL_NAME        Model identifier
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="Email Triage OpenEnv"
LABEL org.opencontainers.image.description="Real-world email triage environment for agent benchmarking"
LABEL org.opencontainers.image.version="1.0.0"

# System dependencies (minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY my_env_v4.py  .
COPY app.py        .
COPY inference.py  .
COPY openenv.yaml  .
COPY README.md     .

# Default task
ENV MY_ENV_V4_TASK=email_urgent

# HF Spaces uses port 7860
EXPOSE 7860

# Health check so orchestrators know when the server is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -sf http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
