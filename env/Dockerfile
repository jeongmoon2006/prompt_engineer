# Build context is the repository root:  docker build -f env/Dockerfile .
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
WORKDIR /app
COPY env/pyproject.toml /app/env/pyproject.toml
RUN uv pip install --system --no-cache -r /app/env/pyproject.toml
COPY shared/ /app/shared/
COPY env/ /app/env/
EXPOSE 8002
CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "8002"]
