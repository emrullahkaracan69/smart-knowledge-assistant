"""
Loads documents from various file formats and splits them into chunks.
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

class DocumentLoader:
    """
    Handles loading and splitting of documents from the specified directory.
    It ensures that the output is a list of LangChain Document objects,
    ready to be ingested by a vector store.
    """
    def __init__(self):
        """Initializes the loader with a text splitter configured from config.py."""
        self.documents_dir = Path(DOCUMENTS_DIR)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True, # Helps in identifying chunk order
        )

    def _load_single_document(self, file_path: Path) -> list[Document]:
        """
        Loads a single document based on its file extension.
        Uses appropriate loaders for PDF, TXT, and MD files.
        """
        extension = file_path.suffix.lower()
        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif extension == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif extension == ".md":
            loader = UnstructuredMarkdownLoader(str(file_path))
        else:
            # Skip unsupported file types gracefully
            return []
        
        # loader.load() returns a list of Document objects
        return loader.load()

    def load_all_documents(self) -> list[Document]:
        """
        Loads all supported documents from the documents directory,
        and then splits them into smaller chunks.
        
        Returns:
            A list of chunked Document objects.
        """
        all_docs = []
        
        # Find all files in the directory with supported extensions
        files_to_load = [
            p for p in self.documents_dir.glob("**/*") 
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        if not files_to_load:
            return []

        # Load content from each file into Document objects
        for file_path in files_to_load:
            docs_from_file = self._load_single_document(file_path)
            if docs_from_file:
                all_docs.extend(docs_from_file)

        if not all_docs:
            return []

        # CRITICAL STEP: Split the loaded Document objects into smaller chunks
        # This method correctly handles Document objects and returns a list of them.
        chunked_docs = self.text_splitter.split_documents(all_docs)
        
        return chunked_docs