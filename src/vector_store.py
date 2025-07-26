
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
from src.config import EMBEDDING_MODEL_NAME, DATA_DIR

PERSIST_DIRECTORY = DATA_DIR / "chroma_db"

class VectorStoreManager:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        
        client_settings = Settings(anonymized_telemetry=False)

        self.db = Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=self.embedding_function,
            client_settings=client_settings
        )

    def get_vector_store(self):
        return self.db