# Build context is the repository root:  docker build -f harness/Dockerfile .
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
WORKDIR /app
COPY harness/pyproject.toml /app/harness/pyproject.toml
RUN uv pip install --system --no-cache -r /app/harness/pyproject.toml
COPY shared/ /app/shared/
COPY harness/ /app/harness/
EXPOSE 8001
CMD ["uvicorn", "harness.server:app", "--host", "0.0.0.0", "--port", "8001"]
