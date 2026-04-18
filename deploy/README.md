# 1C Templates MCP - развертывание

Готовый образ с Docker Hub: `desko77/1c-templates-mcp:latest`

## Требования

- Docker 20.10+ и Docker Compose v2
- Для GPU-режима: NVIDIA GPU + nvidia-container-toolkit

## Быстрый старт (CPU)

```bash
# 1. Скопировать пример переменных (опционально - значения по умолчанию уже заданы)
cp .env.example .env

# 2. Запустить
docker compose up -d

# 3. Проверить
docker compose logs -f
```

Веб-UI: http://localhost:8004
MCP endpoint: http://localhost:8004/mcp

## Запуск с GPU

```bash
docker compose --profile gpu up -d
```

## Управление

```bash
# Логи в реальном времени
docker compose logs -f

# Остановить
docker compose down

# Обновить образ до последней версии
docker compose pull
docker compose up -d

# Полный сброс (удалит индекс и кеш)
docker compose down -v
docker compose up -d
```

## Настройка

Все переменные окружения описаны в `.env.example`. Ключевые:

- `EMBEDDING_PROVIDER` - `local` (по умолчанию), `openai` или `auto`
- `EMBEDDING_MODEL` - модель для локальных эмбеддингов
- `HTTP_PORT` - порт на хосте (по умолчанию 8004)

### Первая индексация

Модель `ai-forever/ru-en-RoSBERTa` (по умолчанию) при первом запуске на CPU индексирует базу 5-10 минут.
Для ускорения можно использовать `intfloat/multilingual-e5-small` - в `.env` поменять `EMBEDDING_MODEL`.

### Внешний API (LM Studio / Ollama / OpenAI)

1. В `.env` раскомментировать блок `OPENAI_*` и задать значения
2. В `docker-compose.yml` раскомментировать соответствующие строки в `environment`
3. Поменять `EMBEDDING_PROVIDER=openai` или `auto`
4. Перезапустить: `docker compose up -d`

## Данные

Индекс и кеш хранятся в именованном volume `app_data`. Удаляется через `docker compose down -v`.
