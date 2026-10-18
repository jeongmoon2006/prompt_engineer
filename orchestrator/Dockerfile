# Build context is the repository root:  docker build -f orchestrator/Dockerfile .
# CPU-capable by default; GPU access is granted at run time via docker-compose.gpu.yml.
# Torch wheels come from the CUDA 12.4 index, which supports Turing (cc 7.5).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
WORKDIR /app
COPY orchestrator/pyproject.toml /app/orchestrator/pyproject.toml
RUN uv pip install --system --no-cache \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      -r /app/orchestrator/pyproject.toml
COPY shared/ /app/shared/
COPY orchestrator/ /app/orchestrator/
EXPOSE 8000
CMD ["uvicorn", "orchestrator.server:app", "--host", "0.0.0.0", "--port", "8000"]
