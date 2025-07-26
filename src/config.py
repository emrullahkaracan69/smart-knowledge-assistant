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
# Try to get API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# If not found in env, try Streamlit secrets (will be handled in app.py)
if not GROQ_API_KEY:
    try:
        import streamlit as st
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
    except:
        # If streamlit is not available or secrets not accessible, leave as None
        pass

# Model configurations
GROQ_MODELS = {
    "llama3-8b-8192": "Llama 3 8B (Balanced)",
    "llama3-70b-8192": "Llama 3 70B (Most Powerful)", 
    "distil-whisper-large-v3-en": "Gemma 7B (Fastest & Most Stable)",
    "mixtral-8x7b-32768": "Mixtral 8x7B (Good for long texts)"
}

# Text chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# RAG settings
TOP_K_RESULTS = 5
TEMPERATURE = 0.3
MAX_ANSWER_LENGTH = 200

# Streamlit settings
PAGE_TITLE = "Smart Knowledge Assistant"
PAGE_ICON = "🧠"

# Supported file formats
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.md']

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)