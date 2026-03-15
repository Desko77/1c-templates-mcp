"""ChromaDB semantic search engine. Derived index from SQLite (source of truth)."""

import logging
import re
import shutil
from pathlib import Path

import chromadb
import numpy as np

from app.config import (
    CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, MAX_BATCH_SIZE,
    MODEL_CACHE_PATH, OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL,
    RESET_CACHE,
)

# Module-level state
model = None
collection = None
_client = None


def _normalize(text: str) -> str:
    """Remove punctuation, collapse whitespace."""
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _init_embedding_model():
    """Try OpenAI-compatible API, fallback to local SentenceTransformer."""
    import os

    api_base = OPENAI_API_BASE
    if not api_base:
        if os.path.exists('/.dockerenv'):
            api_base = 'http://host.docker.internal:1234'
        else:
            api_base = 'http://localhost:1234'

    openai_url = f"{api_base.rstrip('/')}/v1"
    api_key = OPENAI_API_KEY

    # Try OpenAI-compatible API
    try:
        import openai
        logging.info(f"Trying OpenAI-compatible API at {openai_url}...")
        client = openai.OpenAI(base_url=openai_url, api_key=api_key)
        api_model_name = OPENAI_MODEL if OPENAI_MODEL else EMBEDDING_MODEL
        client.embeddings.create(input=["test"], model=api_model_name)
        logging.info(f"Connected to API, model: {api_model_name}")

        class OpenAIEncoder:
            def __init__(self, cl, mn):
                self.client = cl
                self.model_name = mn

            def encode(self, texts, show_progress_bar=False):
                response = self.client.embeddings.create(input=texts, model=self.model_name)
                return np.array([item.embedding for item in response.data])

        return OpenAIEncoder(client, api_model_name)

    except Exception as e:
        logging.warning(f"API unavailable ({e}), falling back to local SentenceTransformer")

    # Fallback: local model
    import torch
    from sentence_transformers import SentenceTransformer

    if RESET_CACHE and MODEL_CACHE_PATH.exists():
        shutil.rmtree(MODEL_CACHE_PATH)
        logging.info("Cleared model cache")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip
        logging.info(f"GPU detected: {gpu_name} ({'ROCm' if is_rocm else 'CUDA'})")
    else:
        logging.warning("No GPU found, using CPU (slow)")

    logging.info(f"Loading local model '{EMBEDDING_MODEL}' on {device}...")
    m = SentenceTransformer(EMBEDDING_MODEL, device=device)
    m.encode(["warmup"])
    logging.info(f"Model loaded on {device}")
    return m


def init_search_engine(force_reindex: bool = False):
    """Initialize embedding model and ChromaDB. Optionally reindex from SQLite."""
    global model, collection, _client

    model = _init_embedding_model()

    chroma_path = Path(CHROMA_DB_PATH)
    if force_reindex and chroma_path.exists():
        logging.info("Force reindex: clearing ChromaDB")
        _clean_directory(chroma_path)

    chroma_path.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(chroma_path))
    collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    logging.info(f"ChromaDB ready: {collection.count()} vectors")


def reindex_all(templates: list[dict]):
    """Full reindex from SQLite data."""
    global collection, _client

    # Drop and recreate collection
    _client.delete_collection(COLLECTION_NAME)
    collection = _client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if not templates:
        logging.warning("No templates to index")
        return

    docs, metas, ids = [], [], []
    total = 0

    for tpl in templates:
        norm = _normalize(tpl['description'] or '')
        if not norm:
            continue
        docs.append(norm)
        metas.append({
            "template_id": str(tpl['id']),
            "description": tpl['description'],
            "code": tpl['code'],
        })
        ids.append(str(tpl['id']))

        if len(docs) >= MAX_BATCH_SIZE:
            _add_batch(docs, metas, ids)
            total += len(docs)
            docs, metas, ids = [], [], []

    if docs:
        _add_batch(docs, metas, ids)
        total += len(docs)

    logging.info(f"Indexed {total} templates in ChromaDB")


def _add_batch(docs, metas, ids):
    try:
        embeddings = model.encode(docs, show_progress_bar=False)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=docs,
            metadatas=metas,
            ids=ids
        )
    except Exception as e:
        logging.error(f"Batch indexing error: {e}")


def index_template(tpl: dict):
    """Index a single template."""
    norm = _normalize(tpl['description'] or '')
    if not norm:
        return
    try:
        embedding = model.encode([norm], show_progress_bar=False)
        collection.upsert(
            embeddings=embedding.tolist(),
            documents=[norm],
            metadatas=[{
                "template_id": str(tpl['id']),
                "description": tpl['description'],
                "code": tpl['code'],
            }],
            ids=[str(tpl['id'])]
        )
    except Exception as e:
        logging.warning(f"Failed to index template {tpl['id']}: {e}")


def update_index(tpl: dict):
    """Update index for an existing template."""
    index_template(tpl)  # upsert handles both create and update


def delete_index(template_id: int):
    """Remove template from ChromaDB."""
    try:
        collection.delete(ids=[str(template_id)])
    except Exception as e:
        logging.warning(f"Failed to delete template {template_id} from index: {e}")


def semantic_search(query: str, n_results: int = 5) -> list[dict]:
    """Hybrid search: vector + full-text, word-count adaptive."""
    norm = _normalize(query)
    if not norm or collection is None or collection.count() == 0:
        return []

    word_count = len(norm.split())
    results = []
    seen_ids = set()

    def _vector():
        nonlocal results, seen_ids
        emb = model.encode([norm])
        res = collection.query(
            query_embeddings=emb.tolist(),
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        if res and res.get('ids') and res['ids'][0]:
            for i, doc_id in enumerate(res['ids'][0]):
                if doc_id not in seen_ids:
                    results.append(res['metadatas'][0][i])
                    seen_ids.add(doc_id)

    def _fulltext():
        nonlocal results, seen_ids
        res = collection.get(
            where_document={"$contains": norm},
            limit=n_results + 1,
            include=["metadatas", "documents"]
        )
        if res and res.get('ids'):
            for i, doc_id in enumerate(res['ids']):
                if doc_id not in seen_ids:
                    results.append(res['metadatas'][i])
                    seen_ids.add(doc_id)

    if word_count == 1:
        _fulltext()
        _vector()
    elif word_count < 4:
        _vector()
        _fulltext()
    else:
        _vector()

    return results


def _clean_directory(path: Path):
    if not path.is_dir():
        return
    for item in path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            logging.error(f"Failed to delete {item}: {e}")
