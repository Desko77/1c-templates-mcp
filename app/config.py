import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)

# --- Paths ---
APP_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv('DATA_DIR', '/app/data'))
MODEL_CACHE_PATH = APP_ROOT / 'model_cache'
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_PATH)

# SQLite: seed DB baked into image, runtime copy in DATA_DIR
BUNDLED_DB_PATH = APP_ROOT / 'templates.db'
TEMPLATES_DB_PATH = Path(os.getenv('TEMPLATES_DB_PATH', str(DATA_DIR / 'templates.db')))

# ChromaDB
CHROMA_DB_PATH = Path(os.getenv('CHROMA_DB_PATH', str(DATA_DIR / 'chroma_db')))
COLLECTION_NAME = "templates_collection"

# --- Server ---
SERVER_NAME = "template-search-mcp"
HTTP_PORT = int(os.getenv('HTTP_PORT', '8004'))

# --- Embedding ---
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'lm-studio')

# --- Flags ---
RESET_CHROMA = os.getenv('RESET_CHROMA', 'false').lower() == 'true'
RESET_CACHE = os.getenv('RESET_CACHE', 'false').lower() == 'true'
USESSE = os.getenv('USESSE', 'false').lower() == 'true'
TRANSPORT = "sse" if USESSE else "streamable-http"

# --- Batch ---
MAX_BATCH_SIZE = 100

logging.info(f"Config: port={HTTP_PORT}, transport={TRANSPORT}, db={TEMPLATES_DB_PATH}, chroma={CHROMA_DB_PATH}")
