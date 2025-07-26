"""
Core RAG (Retrieval-Augmented Generation) engine for the application.
"""
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.vector_store import VectorStoreManager
from src.config import GROQ_API_KEY, GROQ_MODELS, TOP_K_RESULTS, TEMPERATURE

class RAGEngine:
    """
    Manages the entire RAG pipeline from question to answer.
    """
    def __init__(self):
        """Initializes the RAG engine with a vector store and a default model."""
        # Check if API key exists
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please check your environment variables or Streamlit secrets.")
            
        self.vector_store = VectorStoreManager().get_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
        self.groq_api_key = GROQ_API_KEY
        
        # Default model selection
        self.current_model = list(GROQ_MODELS.keys())[0]
        self.prompt_template = self._create_prompt_template()
        
        print(f"RAG Engine initialized with API key: {self.groq_api_key[:10]}...")

    @staticmethod
    def _create_prompt_template():
        """Creates the prompt template for the language model."""
        template = """Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    @staticmethod
    def _format_docs(docs):
        """Formats the retrieved documents into a single string."""
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content[:500] for doc in docs)  # Limit each doc to 500 chars

    def _test_groq_connection(self):
        """Test if Groq API is working."""
        try:
            llm = ChatGroq(
                temperature=0,
                groq_api_key=self.groq_api_key,
                model_name="gemma-7b-it",  # Use the most stable model for testing
                max_tokens=50
            )
            response = llm.invoke("Say 'Hello, Groq is working!'")
            print(f"Groq test successful: {response.content}")
            return True
        except Exception as e:
            print(f"Groq test failed: {str(e)}")
            return False

    def answer_question(self, question: str):
        """
        Answers a user's question using the RAG pipeline.
        
        Args:
            question: The user's question.

        Returns:
            A dictionary containing the answer, sources, and context.
        """
        # First, test Groq connection
        if not self._test_groq_connection():
            return {
                "answer": "❌ Cannot connect to Groq API. Please check:\n1. Your GROQ_API_KEY is correct\n2. You have internet connection\n3. Your Groq account has available credits",
                "sources": [],
                "context_used": []
            }
        
        try:
            # Retrieve relevant documents
            print(f"Retrieving documents for question: {question}")
            retrieved_docs = self.retriever.invoke(question)
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            # If no documents found, return a message
            if not retrieved_docs:
                return {
                    "answer": "📄 No documents found. Please upload documents first before asking questions.",
                    "sources": [],
                    "context_used": []
                }
            
            # Format context
            context = self._format_docs(retrieved_docs)
            print(f"Context length: {len(context)} characters")
            
            # Try simple approach first
            try:
                print(f"Using model: {self.current_model}")
                llm = ChatGroq(
                    temperature=TEMPERATURE,
                    groq_api_key=self.groq_api_key,
                    model_name=self.current_model,
                    max_tokens=500  # Reduced for stability
                )
                
                # Use a simple prompt
                prompt = f"Based on this context:\n\n{context}\n\nAnswer this question: {question}\n\nAnswer:"
                
                print("Sending request to Groq...")
                response = llm.invoke(prompt)
                answer = response.content
                print(f"Received answer: {answer[:100]}...")
                
            except Exception as e:
                print(f"Error with model {self.current_model}: {str(e)}")
                
                # Try with the most stable model
                if self.current_model != "gemma-7b-it":
                    print("Trying with gemma-7b-it...")
                    llm = ChatGroq(
                        temperature=0.3,
                        groq_api_key=self.groq_api_key,
                        model_name="gemma-7b-it",
                        max_tokens=300
                    )
                    response = llm.invoke(f"Question: {question}\n\nContext: {context[:1000]}\n\nBrief answer:")
                    answer = response.content
                else:
                    raise e
            
            # Prepare sources and context
            sources = sorted(list(set(doc.metadata.get("source", "Unknown") for doc in retrieved_docs)))
            context_used = [
                {"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content[:500]}
                for doc in retrieved_docs[:3]  # Limit to 3 documents
            ]

            return {
                "answer": answer,
                "sources": sources,
                "context_used": context_used
            }
            
        except Exception as e:
            error_details = str(e)
            print(f"Final error in answer_question: {error_details}")
            
            # Check for specific error types
            if "api_key" in error_details.lower():
                return {
                    "answer": "🔑 API Key Error: Please check that your GROQ_API_KEY is correctly set in Streamlit secrets.",
                    "sources": [],
                    "context_used": []
                }
            elif "rate" in error_details.lower() or "limit" in error_details.lower():
                return {
                    "answer": "⏱️ Rate limit reached. Please wait a moment and try again, or use a different model.",
                    "sources": [],
                    "context_used": []
                }
            else:
                return {
                    "answer": f"❌ Error: {error_details}\n\nPlease try:\n1. Using a different model\n2. Asking a shorter question\n3. Checking your internet connection",
                    "sources": [],
                    "context_used": []
                }