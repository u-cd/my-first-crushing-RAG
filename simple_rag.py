#!/usr/bin/env python3
"""
Simple command-line interface for the RAG system
No Streamlit required - pure Python interaction
Version: Docker Development Mode
"""

import os
import sys
from pathlib import Path
from typing import List

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_system import RAGSystem


class SimpleRAGInterface:
    """Simple command-line interface for RAG system."""
    
    def __init__(self):
        self.rag = None
        self.setup_rag_system()
    
    def setup_rag_system(self):
        """Initialize the RAG system."""
        print("🚀 Initializing RAG System...")
        print("📄 Processing documents folder - creating fresh chunks and vectors...")
        
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
            print("✅ RAG system initialized with fresh document processing!")
        except Exception as e:
            print(f"❌ Error initializing RAG system: {str(e)}")
            sys.exit(1)
    
    def check_system_status(self):
        """Check if the RAG system is ready for queries."""
        if not self.rag.qa_chain:
            print("❌ No documents loaded. Please add documents to the 'documents' folder and restart.")
            return False
        return True
    
    def ask_question(self, question: str, show_sources: bool = True):
        """Ask a question and get an answer."""
        try:
            if show_sources:
                result = self.rag.generate_answer(question)
                
                # Show the actual prompt sent to LLM
                if 'actual_prompt' in result:
                    print(f"\n📝 Actual Prompt Sent to LLM:")
                    print("=" * 50)
                    print(result['actual_prompt'])
                    print("=" * 50)
                
                print(f"\n🤖 Answer: {result['answer']}")
                
            else:
                answer = self.rag.ask(question)
                print(f"\n🤖 Answer: {answer}")
        
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
    
    def interactive_mode(self):
        """Start interactive question-answering mode."""
        print("\n💬 Interactive Mode - Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                self.ask_question(question)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    """Main function to run the simple RAG interface."""
    print("🤖 Simple RAG System - Command Line Interface")
    print("=" * 50)
    
    # Initialize interface (auto-processes documents folder)
    interface = SimpleRAGInterface()
    
    # Check if system is ready
    if not interface.check_system_status():
        print("\n� To use the system:")
        print("1. Add your documents (PDF, TXT, MD) to the 'documents' folder")
        print("2. Restart the application")
        print("\nDocuments folder location: ./documents/")
        return
    
    # System is ready - start interactive mode
    print("\n✅ System ready! Documents loaded successfully.")
    interface.interactive_mode()


if __name__ == "__main__":
    main()
