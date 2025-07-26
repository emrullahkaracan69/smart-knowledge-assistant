"""
Core RAG (Retrieval-Augmented Generation) engine for the application.
"""
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
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in docs)

    def _create_llm(self):
        """Creates the language model with error handling."""
        try:
            # Create LLM with current model
            llm = ChatGroq(
                temperature=TEMPERATURE,
                groq_api_key=self.groq_api_key,
                model_name=self.current_model,
                max_tokens=1024,  # Explicitly set max tokens
                timeout=30,  # Add timeout
                max_retries=2  # Add retries
            )
            return llm
        except Exception as e:
            print(f"Error creating LLM with model {self.current_model}: {e}")
            # Fallback to a more stable model
            if self.current_model != "gemma-7b-it":
                print("Falling back to gemma-7b-it model")
                self.current_model = "gemma-7b-it"
                return ChatGroq(
                    temperature=TEMPERATURE,
                    groq_api_key=self.groq_api_key,
                    model_name="gemma-7b-it",
                    max_tokens=1024,
                    timeout=30,
                    max_retries=2
                )
            raise

    def _create_rag_chain(self):
        """Creates the full RAG chain with error handling."""
        llm = self._create_llm()
        
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
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(question)
            
            # If no documents found, return a message
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find any relevant information in the uploaded documents. Please make sure you have uploaded documents and they contain information related to your question.",
                    "sources": [],
                    "context_used": []
                }
            
            # Try to generate answer with the RAG chain
            try:
                rag_chain = self._create_rag_chain()
                answer = rag_chain.invoke(question)
            except Exception as e:
                print(f"Error with RAG chain, trying direct LLM approach: {e}")
                
                # Fallback: Use direct message approach
                llm = self._create_llm()
                context = self._format_docs(retrieved_docs)
                
                messages = [
                    SystemMessage(content="You are a helpful assistant that answers questions based on provided context."),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context provided:")
                ]
                
                try:
                    response = llm.invoke(messages)
                    answer = response.content
                except Exception as e2:
                    print(f"Direct LLM approach also failed: {e2}")
                    answer = "I'm sorry, I encountered an error while processing your question. Please try again with a different model or a simpler question."
            
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
            print(f"Error in answer_question: {e}")
            return {
                "answer": f"I encountered an error while processing your request. Please try selecting a different model or simplifying your question. Error: {str(e)}",
                "sources": [],
                "context_used": []
            }