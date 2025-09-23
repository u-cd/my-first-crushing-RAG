"""
Simple demo script to test the RAG system with sample documents
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import rag_system
sys.path.append(str(Path(__file__).parent))

from rag_system import RAGSystem


def create_sample_documents():
    """Create sample documents for testing."""
    
    # Create a documents directory
    docs_dir = Path("sample_documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Sample document 1: About RAG
    doc1_content = """
    Retrieval-Augmented Generation (RAG) Overview
    
    Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models 
    with external knowledge bases to provide more accurate and up-to-date information.
    
    The RAG process consists of three main steps:
    
    1. Indexing: Documents are loaded, chunked, embedded, and stored in a vector database.
    2. Retrieval: When a user submits a query, the system searches for the most relevant documents.
    3. Generation: The retrieved documents are used as context to generate a response.
    
    Key benefits of RAG include:
    - Reduced hallucinations
    - Access to current data
    - Increased trust and transparency
    - Lower costs compared to fine-tuning
    - Enhanced customization
    
    RAG is particularly useful for knowledge-intensive tasks where accuracy and factual correctness 
    are important.
    """
    
    # Sample document 2: About Vector Databases
    doc2_content = """
    Vector Databases in RAG Systems
    
    Vector databases are specialized databases designed to store and query high-dimensional vectors 
    efficiently. In RAG systems, they play a crucial role in the retrieval process.
    
    How Vector Databases Work:
    1. Documents are converted into numerical representations (embeddings) using embedding models
    2. These embeddings are stored in the vector database
    3. When a query comes in, it's also converted to an embedding
    4. The database performs similarity search to find the most relevant documents
    
    Popular vector databases include:
    - ChromaDB: Open-source, easy to use
    - Pinecone: Cloud-based, scalable
    - Weaviate: Open-source with GraphQL API
    - Qdrant: High-performance vector search engine
    
    The choice of embedding model is crucial for good retrieval performance. Common models include:
    - sentence-transformers/all-MiniLM-L6-v2
    - OpenAI text-embedding-ada-002
    - Cohere embed models
    """
    
    # Sample document 3: About LLM Integration
    doc3_content = """
    Integrating Large Language Models with RAG
    
    Large Language Models (LLMs) are the generation component of RAG systems. They take the retrieved 
    context and generate human-like responses.
    
    Popular LLMs for RAG:
    - OpenAI GPT models (GPT-3.5, GPT-4)
    - Anthropic Claude
    - Google PaLM/Gemini
    - Open-source models like Llama, Mistral
    
    Integration Patterns:
    1. API-based: Use cloud-hosted models via APIs
    2. Local deployment: Run models locally using frameworks like Ollama
    3. Hybrid: Combine multiple models for different tasks
    
    Best Practices:
    - Use appropriate prompt templates
    - Implement proper context management
    - Handle token limits effectively
    - Monitor and evaluate responses
    
    The quality of RAG responses depends on both the retrieval accuracy and the LLM's ability 
    to synthesize information from the provided context.
    """
    
    # Write documents to files
    with open(docs_dir / "rag_overview.txt", "w", encoding="utf-8") as f:
        f.write(doc1_content)
    
    with open(docs_dir / "vector_databases.txt", "w", encoding="utf-8") as f:
        f.write(doc2_content)
    
    with open(docs_dir / "llm_integration.txt", "w", encoding="utf-8") as f:
        f.write(doc3_content)
    
    return [
        str(docs_dir / "rag_overview.txt"),
        str(docs_dir / "vector_databases.txt"),
        str(docs_dir / "llm_integration.txt")
    ]


def demo_rag_system():
    """Demonstrate the RAG system with sample questions."""
    
    print("🚀 RAG System Demo")
    print("=" * 50)
    
    try:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
            print("Please set your OpenAI API key in a .env file or environment variable.")
            print("You can still run the indexing and retrieval parts without the API key.")
            print()
        
        # Create sample documents
        print("📄 Creating sample documents...")
        doc_paths = create_sample_documents()
        print(f"Created {len(doc_paths)} sample documents")
        print()
        
        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        rag = RAGSystem()
        print("RAG system initialized")
        print()
        
        # Index documents
        print("📊 Indexing documents...")
        rag.index_documents(doc_paths)
        print("Documents indexed successfully")
        print()
        
        # Test retrieval only (works without OpenAI API key)
        print("🔍 Testing document retrieval...")
        test_query = "What are the benefits of RAG?"
        retrieved_docs = rag.retrieve_documents(test_query, k=2)
        
        print(f"Query: {test_query}")
        print(f"Found {len(retrieved_docs)} relevant documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\n  {i}. Content preview: {doc.page_content[:100]}...")
            if doc.metadata:
                print(f"     Source: {doc.metadata.get('source', 'Unknown')}")
        print()
        
        # Test generation (requires OpenAI API key)
        if os.getenv("OPENAI_API_KEY"):
            print("🤖 Testing answer generation...")
            
            questions = [
                "What are the three main steps of RAG?",
                "What are some popular vector databases?",
                "What are the benefits of using RAG?",
                "How do vector databases work in RAG systems?"
            ]
            
            for question in questions:
                print(f"\n❓ Q: {question}")
                try:
                    result = rag.generate_answer(question)
                    print(f"🎯 A: {result['answer']}")
                    
                    if result['source_documents']:
                        print("📚 Sources:")
                        for i, source in enumerate(result['source_documents'], 1):
                            source_file = source['metadata'].get('source', 'Unknown')
                            print(f"   {i}. {Path(source_file).name}")
                
                except Exception as e:
                    print(f"❌ Error generating answer: {str(e)}")
                
                print("-" * 40)
        
        else:
            print("⚠️  Skipping answer generation (OpenAI API key not found)")
            print("To test generation, set your OPENAI_API_KEY and run again.")
        
        print("\n✅ Demo completed successfully!")
        print("\nTo use the RAG system in your own code:")
        print("1. from rag_system import RAGSystem")
        print("2. rag = RAGSystem()")
        print("3. rag.index_documents(['path/to/your/documents'])")
        print("4. answer = rag.ask('your question')")
        
    except Exception as e:
        print(f"❌ Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_rag_system()
