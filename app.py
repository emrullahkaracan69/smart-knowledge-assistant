"""
Professional Document Q&A System - English Optimized Version
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Bu importu RAGEngine'den önceye alalım.
from src.config import PAGE_TITLE, DOCUMENTS_DIR, SUPPORTED_EXTENSIONS
from src.data_loader import DocumentLoader
from src.rag_engine import RAGEngine

import streamlit as st
from pathlib import Path
import time

# Set page config as the very first Streamlit command
st.set_page_config(
    page_title="Smart Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables if not already present."""
    if 'rag_engine' not in st.session_state:
        with st.spinner("Initializing AI models... This may take a moment."):
            st.session_state.rag_engine = RAGEngine()
            st.session_state.document_loader = DocumentLoader()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm ready to answer questions about your documents. Please upload a document to get started."}
        ]

def render_sidebar():
    """Render the document management sidebar."""
    with st.sidebar:
        st.header("📄 Document Management")

        # AI Model Selection Section
        st.subheader("AI Model Configuration")
        from src.config import GROQ_MODELS
        
        selected_model = st.selectbox(
            "Select Language Model",
            options=list(GROQ_MODELS.keys()),
            format_func=lambda x: GROQ_MODELS[x],
            key="model_selector",
            help="All models are free for development use. Select based on your performance requirements."
        )
        
        if 'selected_model' not in st.session_state or st.session_state.selected_model != selected_model:
            st.session_state.selected_model = selected_model
            if 'rag_engine' in st.session_state:
                st.session_state.rag_engine.current_model = selected_model
        
        st.divider()

        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['txt', 'pdf', 'md'],
            accept_multiple_files=True,
            help="Upload one or more documents. New uploads will replace existing ones."
        )

        if uploaded_files:
            if st.button("Process Documents", type="primary", use_container_width=True):
                handle_file_processing(uploaded_files)
        
        st.divider()
        display_current_documents()
        st.divider()
        display_statistics()

        if st.button("Clear All Data", use_container_width=True):
            clear_all_data()

def handle_file_processing(uploaded_files):
    """Clear old files, save new ones, and process them into the vector store."""
    with st.status("Processing documents...", expanded=True) as status:
        status.update(label="Clearing existing documents...")
        for old_file in DOCUMENTS_DIR.glob("*"):
            if old_file.is_file():
                old_file.unlink()
        
        status.update(label="Saving uploaded documents...")
        for uploaded_file in uploaded_files:
            file_path = DOCUMENTS_DIR / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
        
        status.update(label="Loading and processing documents...")
        loader = st.session_state.document_loader
        docs = loader.load_all_documents()
        
        if not docs:
            status.update(label="No content found in documents.", state="error")
            st.warning("Could not extract any content from the uploaded files. Please check the file format and content.")
            return

        status.update(label="Clearing old index from database...")
        # --- DEĞİŞİKLİK 1: `delete_all()` DÜZELTİLDİ ---
        # ChromaDB'de tümünü silmek için önce tüm ID'leri alıp sonra silmek gerekir.
        vector_store = st.session_state.rag_engine.vector_store
        existing_ids = vector_store.get(include=[])['ids']
        if existing_ids:
            vector_store.delete(ids=existing_ids)
        
        status.update(label="Indexing new documents for semantic search...")
        vector_store.add_documents(docs)
        
        status.update(label="Processing complete!", state="complete")
    
    st.success(f"Successfully processed {len(docs)} document chunks.")
    time.sleep(2)
    st.rerun()

def display_current_documents():
    """Display the list of currently active documents."""
    st.subheader("Active Documents")
    current_docs = [f for f in DOCUMENTS_DIR.glob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if current_docs:
        for doc in current_docs:
            st.text(f"📄 {doc.name}")
    else:
        st.info("No documents uploaded yet.")

def display_statistics():
    """Display database statistics."""
    st.subheader("Database Stats")
    # --- DEĞİŞİKLİK 2: `get_document_count()` DÜZELTİLDİ ---
    # Chroma'da döküman sayısını almak için doğru yöntem `_collection.count()` metodudur.
    vector_store = st.session_state.rag_engine.vector_store
    count = vector_store._collection.count()
    st.metric("Indexed Chunks", count)

def clear_all_data():
    """Clear all documents, vector store, and chat history."""
    vector_store = st.session_state.rag_engine.vector_store
    
    # --- DEĞİŞİKLİK 3: `delete_all()` BURADA DA DÜZELTİLDİ ---
    existing_ids = vector_store.get(include=[])['ids']
    if existing_ids:
        vector_store.delete(ids=existing_ids)
    
    for f in DOCUMENTS_DIR.glob("*"):
        if f.is_file():
            f.unlink()
    
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm ready to answer questions about your documents. Please upload a document to get started."}
    ]
    
    st.success("All documents and chat history have been cleared.")
    time.sleep(1)
    st.rerun()

def render_main_content():
    """Render the main chat interface."""
    st.title(f"🧠 {PAGE_TITLE}")
    st.markdown("Ask questions about your uploaded documents and get AI-powered answers.")
    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Sources"):
                    st.write("**Sources used:**")
                    for source in msg["sources"]:
                        st.markdown(f"- `{source}`")
                    
                    if "context_used" in msg and msg["context_used"]:
                        st.divider()
                        st.write("**Retrieved Context:**")
                        for i, chunk in enumerate(msg["context_used"], 1):
                            with st.container():
                                st.info(f"**Chunk {i} from `{chunk.get('source', 'Unknown')}`:**\n\n{chunk.get('content', '')[:300]}...")

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Analyzing documents and generating answer..."):
                start_time = time.time()
                
                rag_engine = st.session_state.rag_engine
                result = rag_engine.answer_question(prompt)
                
                end_time = time.time()
                response_time = f"{end_time - start_time:.2f}s"
            
            message_placeholder.markdown(result["answer"])

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "context_used": result.get("context_used", []),
            "response_time": response_time
        }
        st.session_state.messages.append(assistant_message)

def main():
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()