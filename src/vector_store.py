__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- DÜZELTME SONU ---

"""
Manages the vector store (ChromaDB) for the application.
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

from src.config import EMBEDDING_MODEL_NAME, DATA_DIR

PERSIST_DIRECTORY = DATA_DIR / "chroma_db"

class VectorStoreManager:
    """
    A class to handle the creation, and retrieval of the Chroma vector store.
    """
    def __init__(self):
        """
        Initializes the VectorStoreManager.
        It sets up the embedding function and initializes the Chroma vector store
        from a persistent directory.
        """
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        # Telemetriyi kapatan ve kalıcı depolama ayarı
        client_settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=str(PERSIST_DIRECTORY)
        )

        self.db = Chroma(
            # persist_directory parametresi artık client_settings içinde
            embedding_function=self.embedding_function,
            client_settings=client_settings
        )

    def get_vector_store(self):
        """
        Returns the initialized Chroma vector store instance.

        Returns:
            Chroma: The vector store instance.
        """
        return self.db