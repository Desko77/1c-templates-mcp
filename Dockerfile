FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Monaco BSL console
RUN git clone --depth 1 https://github.com/salexdv/bsl_console.git /app/bsl_console \
    && rm -rf /app/bsl_console/.git

RUN mkdir -p /app/data /app/model_cache

# Seed database and application code
COPY templates.db /app/templates.db
COPY app/ /app/app/

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    TEMPLATES_DB_PATH=/app/data/templates.db \
    CHROMA_DB_PATH=/app/data/chroma_db \
    HTTP_PORT=8004 \
    EMBEDDING_MODEL=intfloat/multilingual-e5-small \
    RESET_CHROMA=false \
    RESET_CACHE=false

EXPOSE 8004

CMD ["python", "-m", "app.main"]
