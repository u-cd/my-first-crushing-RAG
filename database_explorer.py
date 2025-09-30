#!/usr/bin/env python3
"""
Database Explorer - Show all chunks in the RAG database
"""

import os
import sys
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class DatabaseExplorer:
    """Show all chunks in the RAG database."""
    
    def __init__(self):
        self.rag = None
        self.setup_rag_system()
    
    def setup_rag_system(self):
        """Initialize the RAG system."""
        print("🔍 Database Explorer - Initializing RAG System...")
        
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OpenAI API key not found.")
            api_key = input("Enter your OpenAI API key: ").strip()
            if not api_key:
                print("❌ API key required. Exiting.")
                sys.exit(1)
        
        try:
            self.rag = RAGSystem(openai_api_key=api_key)
            print("✅ RAG system initialized!")
        except Exception as e:
            print(f"❌ Error initializing RAG system: {str(e)}")
            sys.exit(1)
    
    def show_all_chunks(self):
        """Show all chunks stored in the database."""
        print("\n📊 ALL CHUNKS IN DATABASE:")
        print("=" * 80)
        
        try:
            # Get all documents from the collection
            collection = self.rag.vector_store.vectorstore._collection
            all_results = collection.get()
            
            print(f"Total chunks in database: {len(all_results['documents'])}")
            print("\n" + "=" * 80)
            
            for i, (doc_id, content, metadata) in enumerate(zip(
                all_results['ids'], 
                all_results['documents'], 
                all_results['metadatas']
            )):
                print(f"\n🔍 Chunk {i+1}:")
                print(f"ID: {doc_id}")
                print(f"Metadata: {metadata}")
                print(f"Content Length: {len(content)} characters")
                print(f"Content: {content}")
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Error exploring chunks: {e}")


def main():
    """Main function."""
    print("�️ RAG Database Explorer - All Chunks")
    print("=" * 50)
    
    explorer = DatabaseExplorer()
    
    if not explorer.rag.qa_chain:
        print("❌ No documents in database. Please add documents first.")
        return
    
    # Show all chunks immediately
    explorer.show_all_chunks()


if __name__ == "__main__":
    main()
