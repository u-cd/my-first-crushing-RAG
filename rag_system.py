import os
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
                
            except Exception as e:
                pass
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        chunks = self.text_splitter.split_documents(documents)
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
        import uuid
        try:
            # Use a unique collection name to avoid conflicts
            collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            self.vectorstore.persist()
        except Exception as e:
            raise
    
    def load_vectorstore(self) -> None:
        """Load existing vector store."""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        except Exception as e:
            raise
        
    def clear_vectorstore(self) -> None:
        """Clear the existing vector store by removing persist directory."""
        import shutil
        import time
        try:
            # First, try to close any existing vectorstore connection
            if self.vectorstore is not None:
                self.vectorstore = None
                time.sleep(0.5)  # Give it a moment to release resources
            
            if Path(self.persist_directory).exists():
                # Try multiple times to handle resource busy errors
                for attempt in range(3):
                    try:
                        shutil.rmtree(self.persist_directory)
                        break
                    except (OSError, PermissionError) as e:
                        if attempt == 2:  # Last attempt
                            return
                        time.sleep(1)  # Wait before retry
            
            # Recreate the directory
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            pass


class RAGSystem:
    """Main RAG system that orchestrates indexing, retrieval, and generation."""
    
    def __init__(self, openai_api_key: str = None, temperature: float = 0.7, max_tokens: int = 500, documents_folder: str = "./documents", auto_initialize: bool = True):
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
        self.documents_folder = Path(documents_folder)
        
        # Create documents folder if it doesn't exist
        self.documents_folder.mkdir(parents=True, exist_ok=True)
        
        # Auto-initialize if requested
        if auto_initialize:
            self.auto_initialize_from_documents()
    
    def index_documents(self, file_paths: List[str]) -> None:
        """Index documents into the vector store (Step 1: Indexing)."""
        # Clear existing vector store to start fresh
        self.vector_store.clear_vectorstore()
        
        # Load and chunk documents
        documents = self.document_processor.load_documents(file_paths)
        if not documents:
            raise ValueError("No documents were loaded")
        
        chunks = self.document_processor.chunk_documents(documents)
        
        # Create fresh vector store
        self.vector_store.create_vectorstore(chunks)
        
        # Initialize QA chain
        self._initialize_qa_chain()
    
    def auto_initialize_from_documents(self) -> None:
        """Automatically initialize RAG system by processing documents folder."""
        # Always process documents folder to create fresh chunks and vectors
        document_files = self.get_documents_from_folder()
        
        if document_files:
            self.index_documents(document_files)
    
    def get_documents_from_folder(self) -> List[str]:
        """Get all supported document files from the documents folder."""
        supported_extensions = ['.txt', '.pdf', '.md', '.doc', '.docx']
        document_files = []
        
        for ext in supported_extensions:
            pattern = f"*{ext}"
            files = list(self.documents_folder.glob(pattern))
            document_files.extend([str(f) for f in files])
        
        return sorted(document_files)
        
    def _initialize_qa_chain(self, k: int = 4) -> None:
        """Initialize the QA chain with retriever."""
        retriever = self.vector_store.vectorstore.as_retriever(search_kwargs={"k": k})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
    
    def generate_answer(self, query: str, include_sources: bool = True) -> Dict[str, Any]:
        """Generate answer using retrieved context (Step 3: Generation)."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please index documents first.")
        
        # Get answer with source documents - this uses the same retriever as the QA chain
        result = self.qa_chain({"query": query})
        
        # Get the same documents that were actually used by the QA chain
        if "source_documents" in result:
            actual_context = "\n\n".join([doc.page_content for doc in result["source_documents"]])
            actual_prompt = self.prompt_template.format(context=actual_context, question=query)
        else:
            actual_prompt = f"No context retrieved for query: {query}"
        
        response = {
            "answer": result["result"],
            "actual_prompt": actual_prompt,
        }
        
        return response
    

def main():
    """Example usage of the RAG system."""
    try:
        # Initialize RAG system
        print("🚀 Initializing RAG System...")
        rag = RAGSystem()
        
        if not rag.qa_chain:
            print("❌ No documents found. Please add documents to the documents/ folder first.")
            return
            
        print("✅ RAG system initialized successfully!")
        print("💬 You can now ask questions about your documents.")
        print("=" * 50)
        
        # Interactive query loop
        while True:
            try:
                # Get query from user
                query = input("\n❓ Your question (or 'quit' to exit): ").strip()
                
                if not query:
                    print("Please enter a question.")
                    continue
                    
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                # Generate answer
                print("\n🔍 Processing your question...")
                result = rag.generate_answer(query)
                
                # Display results
                print(f"\n🤖 Answer: {result['answer']}")
                print(f"\n📝 Actual Prompt Used:")
                print("=" * 50)
                print(result['actual_prompt'])
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing question: {str(e)}")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {str(e)}")

if __name__ == "__main__":
    main()
