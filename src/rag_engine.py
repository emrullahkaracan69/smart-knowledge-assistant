"""
Core RAG (Retrieval-Augmented Generation) engine for the application.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from src.vector_store import VectorStoreManager
from src.config import GROQ_API_KEY, GROQ_MODELS, TOP_K_RESULTS, TEMPERATURE

class RAGEngine:
    """
    Manages the entire RAG pipeline from question to answer.
    """
    def __init__(self):
        """Initializes the RAG engine with a vector store and a default model."""
        self.vector_store = VectorStoreManager().get_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
        self.groq_api_key = GROQ_API_KEY
        
        # Default model selection
        self.current_model = list(GROQ_MODELS.keys())[0]
        self.prompt_template = self._create_prompt_template()

    @staticmethod
    def _create_prompt_template():
        """Creates the prompt template for the language model."""
        template = """You are an intelligent assistant for a Question Answering system.
Use the following retrieved context to answer the question.
If you don't know the answer, just say "I don't have enough information in the provided documents to answer that question."
Keep the answer concise and based ONLY on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    @staticmethod
    def _format_docs(docs):
        """Formats the retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_rag_chain(self):
        """Creates the full RAG chain."""
        # Create LLM with current model
        llm = ChatGroq(
            temperature=TEMPERATURE,
            groq_api_key=self.groq_api_key,
            model_name=self.current_model
        )

        rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | llm
            | StrOutputParser()
        )
        return rag_chain

    def answer_question(self, question: str):
        """
        Answers a user's question using the RAG pipeline.
        
        Args:
            question: The user's question.

        Returns:
            A dictionary containing the answer, sources, and context.
        """
        try:
            # Create RAG chain
            rag_chain = self._create_rag_chain()
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(question)
            
            # Generate answer
            answer = rag_chain.invoke(question)
            
            # Prepare sources and context
            sources = sorted(list(set(doc.metadata.get("source", "Unknown") for doc in retrieved_docs)))
            context_used = [
                {"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content}
                for doc in retrieved_docs
            ]

            return {
                "answer": answer,
                "sources": sources,
                "context_used": context_used
            }
            
        except Exception as e:
            print(f"Error in answer_question: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your request. Error: {str(e)}",
                "sources": [],
                "context_used": []
            }