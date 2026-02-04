
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
# 1. Try local project root
current_dir = Path(__file__).resolve().parent.parent.parent
env_path = current_dir / ".env"

# 2. If not found, try the specific sibling directory for this user
if not env_path.exists():
    env_path = current_dir.parent / "RAG systems using LlamaIndex" / ".env"

print(f"🔍 Loading .env from: {env_path}")
load_dotenv(env_path)

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPO_DIR = DATA_DIR / "repos"
CHROMA_PATH = DATA_DIR / "chroma_db"
CACHE_DIR = BASE_DIR / "cache"
INGESTION_CACHE = CACHE_DIR / "ingestion_cache.json"
QUERY_CACHE = CACHE_DIR / "query_cache.json"

# --- SETTINGS ---
OPENAI_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# --- IGNORED EXTENSIONS ---
IGNORED_EXTENSIONS = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    # Compiled / Binary
    ".pyc", ".o", ".exe", ".dll", ".so", ".dylib",
    # Models / Data
    ".pth", ".pt", ".onnx", ".pkl", ".bin", ".data", ".h5",
    # Git
    ".git",
    # Lock files
    ".lock",
    # Docs - valid for text but maybe not for code logic? Keeping them for now.
    ".pdf", ".docx"
}
