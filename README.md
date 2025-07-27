```markdown
# Smart Knowledge Assistant

A professional document Q&A system powered by Retrieval-Augmented Generation (RAG) technology. Upload documents and ask questions to get AI-powered answers based on your content.

## Overview

This application enables users to upload documents (PDF, TXT, MD) and interact with them through natural language queries. It uses advanced AI models to understand context and provide accurate answers based solely on the uploaded content.

## Technologies Used

### Core Technologies

**Retrieval-Augmented Generation (RAG)**  
RAG combines the power of large language models with information retrieval systems. Instead of relying solely on pre-trained knowledge, RAG first retrieves relevant information from a knowledge base (your documents) and then generates answers based on that retrieved context. This approach ensures accuracy and prevents hallucinations.

**Vector Embeddings**  
The system uses Sentence Transformers (all-MiniLM-L6-v2) to convert text into high-dimensional vectors. These vectors capture semantic meaning, allowing the system to find relevant passages even when exact keywords don't match. For example, a search for "automobile" would also find passages about "cars" or "vehicles".

**ChromaDB**  
A vector database that stores and efficiently searches through document embeddings. It enables fast similarity search across thousands of document chunks, typically returning results in milliseconds.

**Groq LLM API**  
Provides access to state-of-the-art language models including Llama 3 8B and Llama 3 70B. These models generate human-like responses based on the retrieved context, ensuring answers are both accurate and well-articulated.

### Framework and Libraries

- **Streamlit**: Web application framework for creating the user interface
- **LangChain**: Orchestrates the RAG pipeline, managing document loading, text splitting, and chain operations
- **PyPDF**: Extracts text content from PDF documents
- **HuggingFace Transformers**: Provides the embedding model for semantic search

## Architecture

The application follows a modular architecture:

1. **Document Processing**: Documents are loaded, split into chunks, and converted to embeddings
2. **Vector Storage**: Embeddings are stored in ChromaDB for efficient retrieval
3. **Query Processing**: User questions are embedded and similar chunks are retrieved
4. **Answer Generation**: Retrieved context is passed to the LLM to generate accurate answers

## Usage

1. **Upload Documents**: Use the sidebar to upload PDF, TXT, or Markdown files
2. **Process Documents**: Click "Process Documents" to index your content
3. **Ask Questions**: Type questions in the chat interface
4. **View Sources**: Expand the sources section to see which documents were used

## Configuration

Key configuration parameters in `src/config.py`:

- `CHUNK_SIZE`: Size of text chunks (default: 512 tokens)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100 tokens)
- `TOP_K_RESULTS`: Number of relevant chunks to retrieve (default: 5)
- `TEMPERATURE`: LLM creativity setting (default: 0.3)

## Supported Models

- Llama 3 8B: Balanced performance
- Llama 3 70B: Highest accuracy


## Limitations

- Maximum file size: 50MB per document
- Supported formats: PDF, TXT, MD only
- Requires active internet connection for LLM API
- Document storage is temporary on Streamlit Cloud

## Project Structure

```
smart-knowledge-assistant/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── src/
│   ├── config.py         # Configuration settings
│   ├── data_loader.py    # Document loading utilities
│   ├── rag_engine.py     # RAG pipeline implementation
│   └── vector_store.py   # Vector database management
└── data/
    └── documents/        # Uploaded documents directory
```

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Author

**Emrullah Karacan**  
Email: karacanemrullah69@gmail.com

For questions, suggestions, or issues, please feel free to reach out via email or create an issue on GitHub.
```