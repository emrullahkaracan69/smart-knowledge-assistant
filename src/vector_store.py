"""
Manages the vector store (ChromaDB) for the application.
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME, DATA_DIR

# ChromaDB veritabanının saklanacağı kalıcı dizin
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
        # Gömme (embedding) modelini yükle
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Eğer GPU'nuz yoksa veya kullanmak istemiyorsanız 'cpu' en güvenlisidir
        )

        # Kalıcı depolamadan ChromaDB'yi yükle
        self.db = Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=self.embedding_function
        )

    def get_vector_store(self):
        """
        Returns the initialized Chroma vector store instance.

        Returns:
            Chroma: The vector store instance.
        """
        return self.db
