FROM python:3.11.13 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app


RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt
FROM python:3.11.13-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .
# CMD ["/app/.venv/bin/fastapi", "run"]
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]