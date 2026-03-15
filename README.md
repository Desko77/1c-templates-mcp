# 1c-templates-mcp

MCP-сервер с семантическим поиском по шаблонам кода 1С (BSL). 2200+ шаблонов из сообщества, CRUD веб-интерфейс с Monaco Editor, ChromaDB + embeddings для поиска по смыслу.

<!-- screenshot -->

## Возможности

- **Семантический поиск** - гибридный (vector + full-text) поиск шаблонов кода на русском языке
- **6 MCP-инструментов** - поиск, просмотр, создание, редактирование, удаление шаблонов
- **Веб-интерфейс** - полный CRUD с Monaco Editor и подсветкой BSL-синтаксиса
- **2257 шаблонов** - предустановленная база шаблонов кода 1С
- **Гибкие embeddings** - OpenAI-совместимый API или локальная модель SentenceTransformer
- **Docker** - готовый docker-compose для быстрого запуска

## Быстрый старт

```bash
git clone https://github.com/<your-username>/1c-templates-mcp.git
cd 1c-templates-mcp
docker compose up -d
```

Сервер доступен:
- Веб-интерфейс: `http://localhost:8004`
- MCP endpoint: `http://localhost:8004/mcp` (POST, Streamable HTTP)

### Подключение к Claude Code

```json
{
  "mcpServers": {
    "1c-templates-mcp": {
      "type": "url",
      "url": "http://localhost:8004/mcp"
    }
  }
}
```

## MCP-инструменты

| Инструмент | Параметры | Описание |
|------------|-----------|----------|
| `templatesearch` | `query: str` | Гибридный семантический + полнотекстовый поиск шаблонов |
| `list_templates` | - | Список всех шаблонов (id, name, description, tags) без кода |
| `get_template` | `template_id: int` | Получить полный шаблон с кодом по ID |
| `add_template` | `name, description, code, tags?` | Добавить новый шаблон |
| `update_template` | `template_id, name?, description?, code?, tags?` | Обновить существующий шаблон |
| `delete_template` | `template_id: int` | Удалить шаблон по ID |

## Веб-интерфейс

| Маршрут | Описание |
|---------|----------|
| `GET /` | Список шаблонов с поиском |
| `GET /new` | Форма создания шаблона (Monaco Editor) |
| `GET /{id}` | Просмотр шаблона |
| `GET /{id}/edit` | Редактирование шаблона |
| `POST /{id}/delete` | Удаление шаблона |

## Конфигурация

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `HTTP_PORT` | `8004` | Порт сервера |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Модель для локальных embeddings |
| `OPENAI_API_BASE` | `http://localhost:1234` | URL OpenAI-совместимого API для embeddings |
| `OPENAI_API_KEY` | `lm-studio` | API-ключ |
| `OPENAI_MODEL` | - | Имя модели на API-сервере (переопределяет EMBEDDING_MODEL) |
| `RESET_CHROMA` | `false` | Пересоздать ChromaDB-индекс при старте |
| `RESET_CACHE` | `false` | Очистить кеш модели при старте |
| `USESSE` | `false` | Использовать SSE-транспорт вместо Streamable HTTP |
| `DATA_DIR` | `/app/data` | Директория для runtime-данных (SQLite, ChromaDB) |

## Архитектура

```
                     MCP Clients (Claude Code, Cursor, ...)
                              |
                         POST /mcp
                              |
                    +---------+---------+
                    |    FastAPI app     |
                    |                   |
                    |  /mcp -> FastMCP  |  6 MCP tools
                    |  /    -> Web UI   |  CRUD + Monaco Editor
                    +----+--------+----+
                         |        |
                    +----+--+  +--+------+
                    | SQLite |  | ChromaDB |
                    | (SoT)  |  | (index)  |
                    +--------+  +----+-----+
                                     |
                              +------+------+
                              | Embeddings  |
                              | OpenAI API  |
                              | or local ST |
                              +-------------+
```

- **SQLite** - источник истины (Source of Truth), хранит шаблоны
- **ChromaDB** - векторный индекс для семантического поиска, производный от SQLite
- **Embeddings** - OpenAI-совместимый API (LM Studio, Ollama) или локальный SentenceTransformer

## Embedding-модели

### OpenAI-совместимый API (рекомендуется)

Если доступен сервер с OpenAI-совместимым API (LM Studio, Ollama, vLLM), сервер автоматически использует его:

```yaml
environment:
  - OPENAI_API_BASE=http://host.docker.internal:1234
  - OPENAI_MODEL=text-embedding-qwen3-embedding-0.6b
```

### Локальная модель (fallback)

Если API недоступен, автоматически скачивается и используется `intfloat/multilingual-e5-small` через SentenceTransformer. Поддерживается GPU (CUDA/ROCm) и CPU.

## Локальный запуск (без Docker)

```bash
pip install -r requirements.txt
python -m app.main
```

Для подсветки BSL в веб-интерфейсе клонировать bsl_console рядом с проектом:

```bash
git clone --depth 1 https://github.com/salexdv/bsl_console.git
```

## Благодарности

- [alonehobo/1c_templates_mcp](https://github.com/alonehobo/1c_templates_mcp) - оригинальный MCP-сервер с базой шаблонов
- [salexdv/bsl_console](https://github.com/salexdv/bsl_console) - Monaco Editor с подсветкой BSL-синтаксиса

## Лицензия

MIT
