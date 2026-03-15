import os
import sys
import chromadb
import shutil
import logging
import re
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import torch
import numpy as np
import openai
import uvicorn
from contextlib import asynccontextmanager


def clean_directory(directory_path):
    """
    Deletes all files and subdirectories within a given directory,
    but not the directory itself, preserving the top-level folder.
    """
    path = Path(directory_path)
    if not path.is_dir():
        # Log a warning if the directory doesn't exist, but don't fail.
        logging.warning(f"Directory to clean not found: {path}")
        return

    for item in path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()  # Deletes files and symlinks
        except Exception as e:
            logging.error(f"Failed to delete {item} during cleanup. Reason: {e}")


# --- Configuration ---
# Load environment variables from .env file for local development
load_dotenv()

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

# --- Configuration ---
# Environment variables for controlling cache and database reset
RESET_CACHE = os.getenv('RESET_CACHE', 'true').lower() == 'true'
RESET_DATABASE = os.getenv('RESET_DATABASE', 'true').lower() == 'true'

# Transport selection via environment variable
# USESSE=true → SSE transport (GET /mcp for stream)
# USESSE=false → Streamable HTTP transport (POST /mcp for requests)
USESSE = os.getenv('USESSE', 'false').lower() == 'true'
TRANSPORT = "sse" if USESSE else "streamable-http"
logging.info(f"Transport mode configured: {TRANSPORT}")


# --- License Key Check ---
# For security, the license key is read from an environment variable.
LICENSE_KEY = os.getenv('LICENSE_KEY')
VALID_LICENSE_KEY = 'fad9f22d-6242-4543-b311-e1973e46cb6b'

if not LICENSE_KEY:
    logging.error("FATAL: LICENSE_KEY environment variable is not set.")
    sys.exit("Exiting: LICENSE_KEY is required.")
elif LICENSE_KEY != VALID_LICENSE_KEY:
    logging.error("FATAL: Invalid LICENSE_KEY.")
    sys.exit("Exiting: Application startup cancelled due to invalid license.")
else:
    logging.info("License key provided and validated.")

# --- Environment-dependent paths ---
APP_ROOT = Path(__file__).parent
MODEL_CACHE_PATH = APP_ROOT / 'model_cache'
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_PATH)
logging.info(f"Using model cache path: {MODEL_CACHE_PATH}")

# --- Template database configuration ---
# By default, store templates.db in the same folder as ChromaDB for easier mounting
TEMPLATES_DB_PATH = os.getenv('TEMPLATES_DB_PATH', '/app/chroma_db/templates.db')
BUNDLED_DB_PATH = '/app/templates.db'
logging.info(f"Using templates database: {TEMPLATES_DB_PATH}")

SERVER_NAME = "template-search-mcp"
HTTP_PORT = int(os.getenv('HTTP_PORT', 8004))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', "intfloat/multilingual-e5-small")
OPENAI_MODEL = os.getenv('OPENAI_MODEL')  # Model to use with OpenAI-compatible API
logging.info(f"Using embedding model: {EMBEDDING_MODEL}")
if OPENAI_MODEL:
    logging.info(f"OpenAI API model override: {OPENAI_MODEL}")
COLLECTION_NAME = "templates_collection"

# --- Model Cache Cleanup ---
# Now controlled by RESET_CACHE environment variable
if RESET_CACHE:
    if MODEL_CACHE_PATH.exists() and MODEL_CACHE_PATH.is_dir():
        logging.info(f"RESET_CACHE is True. Removing model cache at: {MODEL_CACHE_PATH}")
        shutil.rmtree(MODEL_CACHE_PATH)
        logging.info("Model cache removed successfully.")
else:
    logging.info("RESET_CACHE is False. Skipping model cache removal.")


# --- ChromaDB and Model Setup ---

# Define database path (configurable via environment variable)
# Default to /app/chroma_db/chroma_db_templates for Docker, or local path if not in Docker
default_chroma_path = "/app/chroma_db/chroma_db_templates" if os.path.exists('/.dockerenv') else os.path.join(str(APP_ROOT.absolute()), "chroma_db_templates")
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', default_chroma_path)
DB_PATH = CHROMA_DB_PATH
logging.info(f"Using ChromaDB path: {DB_PATH}")

# Remove global ChromaDB client and collection initialization
# (moved to main block)

# --- Database Initialization ---

def initialize_database():
    """
    Initialize the templates database.
    If the database doesn't exist at TEMPLATES_DB_PATH, copy from bundled location.
    """
    try:
        db_path = Path(TEMPLATES_DB_PATH)
        
        # Check if database exists
        if db_path.exists():
            logging.info(f"Database already exists at: {TEMPLATES_DB_PATH}")
            return True
        
        # Database doesn't exist, need to copy from bundled location
        logging.info(f"Database not found at: {TEMPLATES_DB_PATH}")
        logging.info("Attempting to copy bundled database...")
        
        # Check if bundled database exists
        bundled_db = Path(BUNDLED_DB_PATH)
        if not bundled_db.exists():
            logging.error(f"Bundled database not found at: {BUNDLED_DB_PATH}")
            logging.error("Cannot initialize database without bundled copy.")
            return False
        
        # Create parent directories if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {db_path.parent}")
        
        # Copy bundled database to target location
        shutil.copy2(bundled_db, db_path)
        logging.info(f"Successfully copied database from {BUNDLED_DB_PATH} to {TEMPLATES_DB_PATH}")
        
        # Verify the copy was successful
        if not db_path.exists():
            logging.error("Database copy verification failed - file does not exist after copy")
            return False
        
        # Verify the file is readable
        if not os.access(db_path, os.R_OK):
            logging.error("Database copy verification failed - file is not readable")
            return False
        
        logging.info("Database initialization successful")
        return True
        
    except PermissionError as e:
        logging.error(f"Permission denied while initializing database: {e}")
        logging.error(f"Target directory '{db_path.parent}' may not be writable")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during database initialization: {e}")
        return False


# --- Template Database Reading ---

def read_templates_from_db():
    """
    Reads templates from the SQLite database.
    Returns a list of dictionaries with template data.
    """
    templates = []
    
    if not os.path.exists(TEMPLATES_DB_PATH):
        logging.error(f"Templates database not found at: {TEMPLATES_DB_PATH}")
        return templates
    
    try:
        conn = sqlite3.connect(TEMPLATES_DB_PATH)
        cursor = conn.cursor()
        
        # Check if snippets table exists (table is named 'snippets', not 'templates')
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='snippets'")
        if not cursor.fetchone():
            logging.error("Table 'snippets' not found in the database")
            conn.close()
            return templates
        
        # Read all templates from snippets table (only description and code columns)
        cursor.execute("SELECT description, code FROM snippets")
        rows = cursor.fetchall()
        
        for idx, row in enumerate(rows):
            template = {
                'id': str(idx + 1),  # Generate ID based on row number
                'name': None,  # No name column in database
                'description': row[0],
                'code': row[1]
            }
            templates.append(template)
        
        conn.close()
        logging.info(f"Successfully read {len(templates)} templates from database")
        
    except sqlite3.Error as e:
        logging.error(f"Error reading from database: {e}")
    
    return templates


def initialize_embedding_model():
    """
    Tries to connect to an OpenAI-compatible API for embeddings.
    If it fails, it falls back to downloading and using a local SentenceTransformer model.
    """
    api_base = os.getenv('OPENAI_API_BASE')
    if api_base:
        logging.info(f"Using custom API base from OPENAI_API_BASE env var: {api_base}")
    else:
        if os.path.exists('/.dockerenv'):
            api_base = 'http://host.docker.internal:1234'
            logging.info(f"Running in Docker, using default API base: {api_base}")
        else:
            api_base = 'http://localhost:1234'
            logging.info(f"Not in Docker, using default API base: {api_base}")

    OPENAI_API_BASE = f"{api_base.rstrip('/')}/v1"
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'lm-studio')  # Placeholder for local servers

    # Try to use OpenAI-compatible API first
    try:
        logging.info(f"Attempting to connect to OpenAI-compatible API at {OPENAI_API_BASE}...")
        client = openai.OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

        # Use OPENAI_MODEL if set, otherwise fall back to EMBEDDING_MODEL
        api_model_name = OPENAI_MODEL if OPENAI_MODEL else EMBEDDING_MODEL
        
        # Test the connection with a small embedding call to ensure the server and model are ready.
        client.embeddings.create(input=["test"], model=api_model_name)
        logging.info(f"Successfully connected to API. Will use model '{api_model_name}' from the server.")

        class OpenAIEncoder:
            """A wrapper class to make the OpenAI API compatible with SentenceTransformer's encode method."""
            def __init__(self, client, model_name):
                self.client = client
                self.model_name = model_name

            def encode(self, texts, show_progress_bar=False):
                # The show_progress_bar is ignored, but kept for compatibility.
                try:
                    logging.info(f"Generating embeddings for {len(texts)} text(s) via API...")
                    response = self.client.embeddings.create(input=texts, model=self.model_name)
                    embeddings = [item.embedding for item in response.data]
                    logging.info("Embeddings generated successfully via API.")
                    return np.array(embeddings)
                except Exception as e:
                    logging.error(f"Failed to get embeddings from API: {e}")
                    raise  # Re-raise the exception to be handled by the calling code.

        return OpenAIEncoder(client, api_model_name)

    except Exception as api_error:
        logging.warning(f"Could not connect to or use the OpenAI-compatible API: {api_error}")
        logging.warning("Falling back to a local SentenceTransformer model.")

        # --- Fallback to local model ---
        try:
            logging.info("Checking for available GPU and pre-downloading local model...")

            device = 'cpu' # Default to CPU
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip or "rocm" in torch.version.cuda

                if is_rocm:
                    logging.info(f"AMD ROCm GPU detected: {gpu_name}")
                else:
                    logging.info(f"NVIDIA CUDA GPU detected: {gpu_name}")
                logging.info(f"Model will run on device: {device}")
            else:
                logging.warning("--------------------------------------------------------------------------")
                logging.warning("WARNING: No CUDA-enabled GPU found. The application will fall back to CPU.")
                logging.warning("Performance will be significantly slower.")
                logging.warning("--------------------------------------------------------------------------")

            logging.info(f"Loading model '{EMBEDDING_MODEL}' onto '{device}'...")
            model = SentenceTransformer(EMBEDDING_MODEL, device=device)
            model.encode(["test to warm up"])
            logging.info(f"Model '{EMBEDDING_MODEL}' downloaded and loaded successfully on {device}.")
            return model
        except Exception as e:
            logging.error(f"FATAL: Failed to download or initialize local fallback model: {e}")
            sys.exit("Exiting: Critical error during local model initialization.")


# Maximum batch size for ChromaDB.
# A large batch size can cause out-of-memory errors.
# If the process exits silently, try reducing this value.
MAX_BATCH_SIZE = 100


def index_templates(collection):
    """
    Reads templates from the SQLite database and indexes them in ChromaDB.
    This runs on server startup.
    """
    
    logging.info("Starting to index templates from database: %s", TEMPLATES_DB_PATH)
    
    # Read templates from database
    templates = read_templates_from_db()
    
    if not templates:
        logging.warning("No templates found in database. Nothing to index.")
        return

    docs_to_add = []
    metadata_to_add = []
    ids_to_add = []
    total_templates_processed = 0

    try:
        for template in templates:
            try:
                template_id = str(template['id'])
                logging.info("Processing template ID %s", template_id)
                
                # Use the description for embeddings
                description = template['description'] or ""
                
                # Normalize the description for embedding
                # Keep only alphanumeric characters and whitespace
                normalized_description = re.sub(r'[^\w\s]', ' ', description)
                # Collapse multiple whitespace characters into a single space
                normalized_description = re.sub(r'\s+', ' ', normalized_description).strip()
                
                if normalized_description:
                    docs_to_add.append(normalized_description)
                    metadata_to_add.append({
                        "template_id": template_id,
                        "description": template['description'],
                        "code": template['code']
                    })
                    ids_to_add.append(template_id)  # Use template ID as the document ID

                    # If we've reached the batch size limit, process this batch
                    if len(docs_to_add) >= MAX_BATCH_SIZE:
                        try:
                            logging.info(f"Processing batch of {len(docs_to_add)} templates...")
                            logging.info("Generating embeddings...")
                            embeddings = model.encode(docs_to_add, show_progress_bar=True)
                            
                            logging.info("Adding templates and embeddings to the database...")
                            collection.add(
                                embeddings=embeddings.tolist(),
                                documents=docs_to_add,
                                metadatas=metadata_to_add,
                                ids=ids_to_add
                            )
                            total_templates_processed += len(docs_to_add)
                            logging.info(f"Successfully indexed {total_templates_processed} templates so far.")
                            
                            # Clear the batch
                            docs_to_add = []
                            metadata_to_add = []
                            ids_to_add = []
                        except Exception as e:
                            logging.error(f"Error processing batch: {e}. This batch will be skipped.")
                            # Clear the batch to prevent reprocessing
                            docs_to_add = []
                            metadata_to_add = []
                            ids_to_add = []
                else:
                    logging.warning(f"Template ID {template_id} has empty description and was skipped.")
            except Exception as e:
                logging.warning(f"Skipping template ID {template.get('id', 'unknown')} due to error: {e}")
                continue

        # Process any remaining templates
        if docs_to_add:
            try:
                logging.info(f"Processing final batch of {len(docs_to_add)} templates...")
                logging.info("Generating embeddings...")
                embeddings = model.encode(docs_to_add, show_progress_bar=True)
                
                logging.info("Adding templates and embeddings to the database...")
                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=docs_to_add,
                    metadatas=metadata_to_add,
                    ids=ids_to_add
                )
                total_templates_processed += len(docs_to_add)
            except Exception as e:
                logging.error(f"Error processing final batch: {e}. This batch could not be indexed.")
                
        if total_templates_processed > 0:
            logging.info(f"Template indexing completed. Total templates indexed: {total_templates_processed}")
        else:
            logging.info("No valid templates found to index.")
    except Exception as e:
        logging.error(f"Critical error during template indexing: {e}")
        raise


# --- Template Insertion ---

def insert_template_to_db(description: str, code: str):
    """
    Insert a new template into SQLite database and index in vector DB.
    Returns (success: bool, message: str)
    """
    try:
        # Step 1: Insert into SQLite
        conn = sqlite3.connect(TEMPLATES_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("INSERT INTO snippets (description, code) VALUES (?, ?)", (description, code))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logging.info(f"Successfully inserted template into SQLite with ID: {new_id}")
        
        # Step 2: Index in vector database
        try:
            # Normalize the description for embedding (same logic as in index_templates)
            normalized_description = re.sub(r'[^\w\s]', ' ', description)
            normalized_description = re.sub(r'\s+', ' ', normalized_description).strip()
            
            if normalized_description:
                # Generate embedding
                embedding = model.encode([normalized_description], show_progress_bar=False)
                
                # Add to collection
                collection.add(
                    embeddings=embedding.tolist(),
                    documents=[normalized_description],
                    metadatas=[{
                        "template_id": str(new_id),
                        "description": description,
                        "code": code
                    }],
                    ids=[str(new_id)]
                )
                logging.info(f"Successfully indexed template ID {new_id} in vector database")
            else:
                logging.warning(f"Template ID {new_id} has empty normalized description, skipped vector indexing")
                
        except Exception as e:
            logging.warning(f"Failed to index template in vector database: {e}")
            logging.warning("Template was saved to SQLite but not indexed for search")
        
        return True, f"Шаблон успешно добавлен (ID: {new_id})"
        
    except sqlite3.IntegrityError as e:
        logging.error(f"Database integrity error: {e}")
        return False, "Шаблон с похожим описанием уже может существовать"
    except sqlite3.Error as e:
        logging.error(f"Database error during template insertion: {e}")
        return False, f"Ошибка базы данных: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error during template insertion: {e}")
        return False, f"Неожиданная ошибка: {str(e)}"


# --- FastMCP Server Definition ---
mcp = FastMCP()

@mcp.tool()
def templatesearch(query: str) -> str:
    """
    Searches the 1C code template or some additional context for a given query.

    Args:
        query: The search term or question in Russian, describing the desired functionality or some case

    Returns:
        A formatted string with context or code
    """
    global collection
    logging.info("Received query: '%s'", query)

    # Normalize the query to match the indexed content by replacing punctuation with spaces
    normalized_query = re.sub(r'[^\w\s]', ' ', query).strip()
    logging.info("Normalized query: '%s'", normalized_query)

    if not normalized_query:
        return "Query is empty. Please provide a search term."

    word_count = len(normalized_query.split())
    logging.info("Query word count: %d", word_count)

    if collection.count() == 0:
        return "The template database is empty. Please ensure 'templates.db' exists and restart the server to index the templates."

    response_parts = []
    processed_ids = set()

    # --- Vector Search Logic ---
    def perform_vector_search():
        nonlocal response_parts, processed_ids
        logging.info("Performing vector search...")
        query_embedding = model.encode([normalized_query])
        vector_results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5,
            include=["metadatas", "documents", "distances"]
        )

        if vector_results and vector_results.get('documents') and vector_results['documents'][0]:
            for i, doc in enumerate(vector_results['documents'][0]):
                doc_id = vector_results['ids'][0][i]
                if doc_id in processed_ids:
                    continue

                metadata = vector_results['metadatas'][0][i]
                template_result = format_template_result(metadata)
                
                logging.debug(f"--- DIAGNOSTIC: Retrieved template from DB (Vector) for ID: {doc_id} ---")

                response_parts.append(template_result)
                processed_ids.add(doc_id)

    # --- Full-text Search Logic ---
    def perform_fulltext_search():
        nonlocal response_parts, processed_ids
        logging.info("Performing full-text search...")
        fulltext_results = collection.get(
            where_document={"$contains": normalized_query},
            limit=6,
            include=["metadatas", "documents"]
        )

        if fulltext_results and fulltext_results.get('documents'):
            for i, doc_id in enumerate(fulltext_results['ids']):
                if doc_id not in processed_ids:
                    metadata = fulltext_results['metadatas'][i]
                    template_result = format_template_result(metadata)
                    
                    logging.debug(f"--- DIAGNOSTIC: Retrieved template from DB (Full-Text) for ID: {doc_id} ---")

                    response_parts.append(template_result)
                    processed_ids.add(doc_id)

    # --- Main Search Logic based on word count ---
    if word_count == 1:
        # For single-word queries, full-text is often more precise.
        perform_fulltext_search()
        perform_vector_search()
    elif word_count > 1 and word_count < 4:
        # For two-word queries, a hybrid approach is good.
        perform_vector_search()
        perform_fulltext_search()
    else:  # 3 or more words
        # For long queries, vector search is better at capturing semantic meaning.
        perform_vector_search()

    if not response_parts:
        return "No relevant templates were found for your query."

    full_response = "\n---\n".join(response_parts)
    return full_response


def format_template_result(metadata):
    """Formats a template result according to the specification."""
    description = metadata.get('description', 'No description available')
    code = metadata.get('code', 'No code available')
    
    # Since there's no name field, we'll use the first part of description as a title
    title = description.split('.')[0] if description else 'Unknown Template'
    if len(title) > 80:  # Truncate if too long
        title = title[:77] + '...'
    
    return f"**Template:** {title}\n**Description:** {description}\n**Code:**\n```\n{code}\n```"


# --- Startup Hook for Initialization ---

def startup():
    """Initialize database, model, and ChromaDB on server startup."""
    global client, collection, model
    
    logging.info("=== Application Startup ===")
    
    # Database Initialization
    logging.info("Step 1: Initializing templates database...")
    if not initialize_database():
        logging.error("FATAL: Database initialization failed")
        raise RuntimeError("Database initialization failed")
    logging.info("Database initialization completed.")
    
    # Model Initialization
    logging.info("Step 2: Initializing embedding model (API or local)...")
    model = initialize_embedding_model()
    logging.info("Model loaded successfully.")
    
    # ChromaDB Initialization
    logging.info("Step 3: Initializing ChromaDB...")
    if RESET_DATABASE:
        if os.path.exists(DB_PATH):
            logging.info("RESET_DATABASE is True. Clearing existing ChromaDB database.")
            clean_directory(DB_PATH)
            logging.info("Successfully cleared old database.")
        else:
            logging.info(f"RESET_DATABASE is True, but no existing ChromaDB found at '{DB_PATH}'.")
    else:
        logging.info("RESET_DATABASE is False. Using existing database if available.")
    
    logging.info(f"Creating/loading ChromaDB database at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    logging.info("ChromaDB initialization completed.")
    
    # Template Indexing
    if RESET_DATABASE:
        # Only index if RESET_DATABASE is True
        logging.info("Step 4: RESET_DATABASE is True. Starting template indexing...")
        index_templates(collection)
    else:
        # Skip indexing if RESET_DATABASE is False
        if collection.count() > 0:
            logging.info("Step 4: RESET_DATABASE is False. Using existing ChromaDB with %d templates.", collection.count())
        else:
            logging.info("Step 4: RESET_DATABASE is False. ChromaDB is empty but will not be reindexed.")
            logging.warning("WARNING: ChromaDB collection is empty. Set RESET_DATABASE=true to reindex from templates.db")
    
    logging.info("=== Application Ready ===")


# --- Create FastMCP ASGI app ---
# Create MCP app with specified transport
# Using path="" for SSE ensures the endpoint is at /mcp (not /mcp/sse)
logging.info(f"Creating FastMCP HTTP app with transport: {TRANSPORT}")

try:
    # For SSE: Use path="/mcp" to create GET /mcp endpoint, mount at root
    # For Streamable HTTP: Use path="/" to create POST / endpoint, mount at /mcp
    if TRANSPORT == "sse":
        mcp_app = mcp.http_app(transport=TRANSPORT, path="/mcp")
        logging.info(f"FastMCP SSE app created with path='/mcp'")
    else:
        mcp_app = mcp.http_app(transport=TRANSPORT, path="/")
        logging.info(f"FastMCP Streamable HTTP app created with path='/'")
        
except Exception as e:
    logging.error(f"Fatal: Cannot create FastMCP app: {e}")
    sys.exit(1)

# --- Setup custom lifespan for our FastAPI app ---
@asynccontextmanager
async def combined_lifespan(app_instance):
    """Run our startup logic and the mcp_app's lifespan."""
    # Run our own startup
    startup()
    
    # Invoke the mcp_app's lifespan to initialize its task group
    if hasattr(mcp_app, 'router') and hasattr(mcp_app.router, 'lifespan_context'):
        async with mcp_app.router.lifespan_context(mcp_app) as state:
            yield state
    else:
        yield
    
    # Shutdown
    logging.info("Server shutting down...")

# --- Create Combined FastAPI + FastMCP Application ---
# Create main FastAPI app with the combined lifespan
app = FastAPI(
    title="Template Search MCP Server with Web Interface",
    description="MCP server for 1C template search with web form for template submission",
    version="1.0.0",
    lifespan=combined_lifespan
)

# Add web interface routes to the main app
@app.get("/extend", response_class=HTMLResponse)
@app.get("/extend/", response_class=HTMLResponse)
async def extend_form_get():
    """Display the template submission form."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Добавить новый шаблон</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                border-radius: 8px;
                padding: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
            }
            label {
                display: block;
                font-weight: 600;
                color: #333;
                margin-bottom: 8px;
                margin-top: 20px;
            }
            textarea {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: inherit;
                font-size: 14px;
                box-sizing: border-box;
                resize: vertical;
            }
            textarea:focus {
                outline: none;
                border-color: #4CAF50;
            }
            #description {
                min-height: 100px;
            }
            #code {
                min-height: 400px;
                font-family: 'Courier New', Consolas, Monaco, monospace;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 20px;
            }
            button:hover {
                background-color: #45a049;
            }
            .hint {
                color: #666;
                font-size: 13px;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Добавить новый шаблон</h1>
            <p class="subtitle">Отправить новый шаблон кода в базу данных</p>
            
            <form method="POST" action="/extend">
                <label for="description">Описание</label>
                <div class="hint">Краткое описание шаблона (минимум 10 символов)</div>
                <textarea id="description" name="description" placeholder="Введите описание шаблона..." required></textarea>
                
                <label for="code">Код</label>
                <div class="hint">Фрагмент кода (минимум 10 символов)</div>
                <textarea id="code" name="code" placeholder="Введите код здесь..." required></textarea>
                
                <button type="submit">Отправить шаблон</button>
            </form>
        </div>
    </body>
    </html>
    """
    return html


@app.post("/extend", response_class=HTMLResponse)
@app.post("/extend/", response_class=HTMLResponse)
async def extend_form_post(description: str = Form(""), code: str = Form("")):
    """Handle template submission."""
    
    # Validation
    errors = []
    
    if not description or not description.strip():
        errors.append("Требуется описание")
    elif len(description) < 10:
        errors.append("Описание должно содержать не менее 10 символов")
    
    if not code or not code.strip():
        errors.append("Требуется код")
    elif len(code) < 10:
        errors.append("Код должен содержать не менее 10 символов")
    
    # If validation fails, return error page
    if errors:
        error_list = "<br>".join(f"• {err}" for err in errors)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ошибка валидации</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    max-width: 900px;
                    margin: 40px auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #d32f2f;
                    margin-bottom: 20px;
                }}
                .errors {{
                    background-color: #ffebee;
                    border-left: 4px solid #d32f2f;
                    padding: 15px;
                    margin: 20px 0;
                    color: #c62828;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    color: #4CAF50;
                    text-decoration: none;
                    font-weight: 600;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Ошибка валидации</h1>
                <div class="errors">
                    {error_list}
                </div>
                <a href="/extend">← Вернуться к форме</a>
            </div>
        </body>
        </html>
        """
        return html
    
    # Insert template
    success, message = insert_template_to_db(description.strip(), code.strip())
    
    if success:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Успех</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    max-width: 900px;
                    margin: 40px auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #4CAF50;
                    margin-bottom: 20px;
                }}
                .success {{
                    background-color: #e8f5e9;
                    border-left: 4px solid #4CAF50;
                    padding: 15px;
                    margin: 20px 0;
                    color: #2e7d32;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    margin-right: 15px;
                    color: #4CAF50;
                    text-decoration: none;
                    font-weight: 600;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>✓ Успех</h1>
                <div class="success">
                    {message}
                </div>
                <a href="/extend">Добавить еще один шаблон</a>
            </div>
        </body>
        </html>
        """
        return html
    else:
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Ошибка</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    max-width: 900px;
                    margin: 40px auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #d32f2f;
                    margin-bottom: 20px;
                }}
                .error {{
                    background-color: #ffebee;
                    border-left: 4px solid #d32f2f;
                    padding: 15px;
                    margin: 20px 0;
                    color: #c62828;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    color: #4CAF50;
                    text-decoration: none;
                    font-weight: 600;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Ошибка</h1>
                <div class="error">
                    {message}
                </div>
                <a href="/extend">← Вернуться к форме</a>
            </div>
        </body>
        </html>
        """
        return html

# --- Mount FastMCP app into the main FastAPI app ---
# For SSE: mount at root because path is already "/mcp"
# For Streamable HTTP: mount at /mcp
if TRANSPORT == "sse":
    app.mount("/", mcp_app)
    logging.info(f"FastMCP SSE app mounted at / (with internal path /mcp)")
else:
    app.mount("/mcp", mcp_app)
    logging.info(f"FastMCP app mounted at /mcp")
if TRANSPORT == "sse":
    logging.info(f"SSE transport active:")
    logging.info(f"  - SSE stream (GET): http://0.0.0.0:{HTTP_PORT}/mcp")
    logging.info(f"  - SSE messages (POST): http://0.0.0.0:{HTTP_PORT}/mcp/messages/")
else:
    logging.info(f"Streamable HTTP transport active:")
    logging.info(f"  - HTTP endpoint (POST): http://0.0.0.0:{HTTP_PORT}/mcp")
logging.info(f"Web interface available at: http://0.0.0.0:{HTTP_PORT}/extend")


# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        logging.info("=" * 70)
        logging.info(f"Starting {SERVER_NAME} server...")
        logging.info(f"License: VALID")
        logging.info(f"Transport Mode: {TRANSPORT.upper()}")
        logging.info("=" * 70)
        logging.info("Server Configuration:")
        logging.info(f"  Host: 0.0.0.0")
        logging.info(f"  Port: {HTTP_PORT}")
        logging.info("=" * 70)
        logging.info("Available Endpoints:")
        logging.info(f"  Web UI: http://0.0.0.0:{HTTP_PORT}/extend")
        
        if TRANSPORT == "sse":
            logging.info(f"  MCP SSE Stream (GET): http://0.0.0.0:{HTTP_PORT}/mcp")
            logging.info(f"  MCP SSE Messages (POST): http://0.0.0.0:{HTTP_PORT}/mcp/messages/")
        else:
            logging.info(f"  MCP Streamable HTTP (POST): http://0.0.0.0:{HTTP_PORT}/mcp")
        
        logging.info("=" * 70)
        logging.info("Transport Configuration:")
        logging.info(f"  Current: {TRANSPORT.upper()}")
        logging.info("  To switch: Set USESSE=true (SSE) or USESSE=false (Streamable HTTP)")
        logging.info("=" * 70)
        
        # Run the combined FastAPI + FastMCP server
        uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="info")
        
    except Exception as e:
        logging.error(f"FATAL: Application failed to start. Error: {e}")
        sys.exit(1)