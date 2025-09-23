import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading and chunking for the RAG system."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from various file formats."""
        documents = []
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks


class VectorStore:
    """Handles vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create a vector store from documents."""
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vectorstore.persist()
            logger.info(f"Created vector store with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vectorstore(self) -> None:
        """Load existing vector store."""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("Loaded existing vector store")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on the vector store."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def as_retriever(self, k: int = 4):
        """Return the vector store as a retriever."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


class RAGSystem:
    """Main RAG system that orchestrates indexing, retrieval, and generation."""
    
    def __init__(self, openai_api_key: str = None, temperature: float = 0.7, max_tokens: int = 500):
        # Set up OpenAI
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.openai_api_key
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        
        # LLM configuration
        self.llm = OpenAI(
            openai_api_key=self.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Custom prompt template
        self.prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = None
    
    def index_documents(self, file_paths: List[str]) -> None:
        """Index documents into the vector store (Step 1: Indexing)."""
        logger.info("Starting document indexing...")
        
        # Load and chunk documents
        documents = self.document_processor.load_documents(file_paths)
        if not documents:
            raise ValueError("No documents were loaded")
        
        chunks = self.document_processor.chunk_documents(documents)
        
        # Create vector store
        self.vector_store.create_vectorstore(chunks)
        
        # Initialize QA chain
        self._initialize_qa_chain()
        
        logger.info("Document indexing completed successfully")
    
    def load_existing_index(self) -> None:
        """Load an existing vector store index."""
        logger.info("Loading existing index...")
        self.vector_store.load_vectorstore()
        self._initialize_qa_chain()
        logger.info("Index loaded successfully")
    
    def _initialize_qa_chain(self) -> None:
        """Initialize the QA chain with retriever."""
        retriever = self.vector_store.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
    
    def retrieve_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents (Step 2: Retrieval)."""
        if not self.vector_store.vectorstore:
            raise ValueError("Vector store not initialized. Please index documents first.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def generate_answer(self, query: str, include_sources: bool = True) -> Dict[str, Any]:
        """Generate answer using retrieved context (Step 3: Generation)."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please index documents first.")
        
        logger.info(f"Processing query: {query}")
        
        # Get answer with source documents
        result = self.qa_chain({"query": query})
        
        response = {
            "query": query,
            "answer": result["result"],
            "source_documents": []
        }
        
        if include_sources and "source_documents" in result:
            for i, doc in enumerate(result["source_documents"]):
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "index": i + 1
                }
                response["source_documents"].append(source_info)
        
        logger.info("Answer generated successfully")
        return response
    
    def ask(self, question: str) -> str:
        """Simple interface to ask a question and get an answer."""
        result = self.generate_answer(question, include_sources=False)
        return result["answer"]


def main():
    """Example usage of the RAG system."""
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        # Example: Index some documents
        # documents = ["path/to/document1.txt", "path/to/document2.pdf"]
        # rag.index_documents(documents)
        
        # Or load existing index
        # rag.load_existing_index()
        
        # Ask questions
        # question = "What is retrieval-augmented generation?"
        # answer = rag.ask(question)
        # print(f"Q: {question}")
        # print(f"A: {answer}")
        
        print("RAG system initialized successfully!")
        print("To use:")
        print("1. Add your documents and call rag.index_documents([file_paths])")
        print("2. Ask questions using rag.ask('your question')")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
