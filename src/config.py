"""
Project configuration settings - Optimized for English documents
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories  
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Embedding model - Best lightweight model for English
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Groq API Configuration
GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY") or 
    st.secrets.get("GROQ_API_KEY", None) if "secrets" in dir(st) else None
)

# Updated model names to ensure compatibility
GROQ_MODELS = {
    "llama3-8b-8192": "Llama 3 8B (Balanced)",
    "llama3-70b-8192": "Llama 3 70B (Most Powerful)", 
    "gemma-7b-it": "Gemma 7B (Fastest & Most Stable)",
    "mixtral-8x7b-32768": "Mixtral 8x7B (Good for long texts)"
}

# Text chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# RAG settings
TOP_K_RESULTS = 5
TEMPERATURE = 0.3  # Lower temperature for more consistent answers
MAX_ANSWER_LENGTH = 200

# Streamlit settings
PAGE_TITLE = "Smart Knowledge Assistant"
PAGE_ICON = "🧠"

# Supported file formats
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.md']

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)