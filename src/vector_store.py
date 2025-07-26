"""
Vector store management using ChromaDB for document embeddings.
"""

import sys
# SQLite3 compatibility fix for deployment environments
if 'sqlite3' in sys.modules:
    del sys.modules['sqlite3']
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from src.config import EMBEDDING_MODEL_NAME, DATA_DIR

# Try different embedding options based on availability
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = "huggingface"
except ImportError:
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        EMBEDDINGS_AVAILABLE = "sentence_transformer"
    except ImportError:
        # Fallback to a simple embedding
        from langchain.embeddings.base import Embeddings
        from typing import List
        import hashlib
        
        class SimpleEmbeddings(Embeddings):
            """Simple hash-based embeddings as fallback."""
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed documents using simple hash."""
                embeddings = []
                for text in texts:
                    # Create a simple 384-dimensional embedding
                    hash_obj = hashlib.sha384(text.encode())
                    hash_hex = hash_obj.hexdigest()
                    # Convert hex to float values
                    embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, min(768, len(hash_hex)), 2)]
                    # Pad to 384 dimensions if needed
                    while len(embedding) < 384:
                        embedding.append(0.0)
                    embeddings.append(embedding[:384])
                return embeddings
            
            def embed_query(self, text: str) -> List[float]:
                """Embed a query using simple hash."""
                return self.embed_documents([text])[0]
        
        EMBEDDINGS_AVAILABLE = "simple"

PERSIST_DIRECTORY = DATA_DIR / "chroma_db"

class VectorStoreManager:
    """
    Manages the ChromaDB vector store for document embeddings.
    """
    def __init__(self):
        """Initialize the vector store with available embeddings."""
        print(f"Using {EMBEDDINGS_AVAILABLE} embeddings")
        
        if EMBEDDINGS_AVAILABLE == "huggingface":
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif EMBEDDINGS_AVAILABLE == "sentence_transformer":
            self.embedding_function = SentenceTransformerEmbeddings(
                model_name=EMBEDDING_MODEL_NAME
            )
        else:
            # Use simple embeddings as fallback
            self.embedding_function = SimpleEmbeddings()
        
        # Configure ChromaDB settings
        client_settings = Settings(
            anonymized_telemetry=False,
            persist_directory=str(PERSIST_DIRECTORY),
            is_persistent=True
        )

        # Create collection name based on embedding type
        collection_name = f"documents_{EMBEDDINGS_AVAILABLE}"

        try:
            self.db = Chroma(
                persist_directory=str(PERSIST_DIRECTORY),
                embedding_function=self.embedding_function,
                client_settings=client_settings,
                collection_name=collection_name
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            # Try without persistence
            self.db = Chroma(
                embedding_function=self.embedding_function,
                collection_name=collection_name
            )

    def get_vector_store(self):
        """Return the initialized vector store."""
        return self.db