"""
Advanced RAG example with multiple LLM providers and custom configurations
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

from rag_system import RAGSystem


class AdvancedRAGSystem(RAGSystem):
    """Extended RAG system with additional features."""
    
    def __init__(self, llm_provider: str = "openai", **kwargs):
        """
        Initialize advanced RAG system with different LLM providers.
        
        Args:
            llm_provider: "openai", "anthropic", or "local"
            **kwargs: Additional configuration parameters
        """
        self.llm_provider = llm_provider
        
        if llm_provider == "openai":
            super().__init__(**kwargs)
        else:
            # Initialize base components without OpenAI
            from rag_system import DocumentProcessor, VectorStore
            self.document_processor = DocumentProcessor()
            self.vector_store = VectorStore()
            self._setup_custom_llm(llm_provider, **kwargs)
    
    def _setup_custom_llm(self, provider: str, **kwargs):
        """Setup custom LLM provider."""
        if provider == "anthropic":
            self._setup_anthropic(**kwargs)
        elif provider == "local":
            self._setup_local_llm(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _setup_anthropic(self, **kwargs):
        """Setup Anthropic Claude."""
        try:
            from langchain.llms import Anthropic
            api_key = kwargs.get('anthropic_api_key') or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key is required")
            
            self.llm = Anthropic(
                anthropic_api_key=api_key,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens_to_sample=kwargs.get('max_tokens', 500)
            )
            print("✅ Anthropic Claude initialized")
        except ImportError:
            print("❌ Anthropic package not installed. Run: pip install anthropic")
            raise
    
    def _setup_local_llm(self, **kwargs):
        """Setup local LLM using Ollama."""
        try:
            from langchain.llms import Ollama
            model = kwargs.get('model', 'llama2')
            
            self.llm = Ollama(
                model=model,
                temperature=kwargs.get('temperature', 0.7)
            )
            print(f"✅ Local LLM ({model}) initialized")
        except ImportError:
            print("❌ Ollama package not installed. Run: pip install ollama")
            raise
    
    def evaluate_responses(self, test_questions: list, ground_truth: list = None) -> Dict[str, Any]:
        """
        Evaluate RAG system performance on a set of test questions.
        
        Args:
            test_questions: List of questions to test
            ground_truth: Optional list of expected answers for comparison
        
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'questions': [],
            'answers': [],
            'retrieval_scores': [],
            'response_times': []
        }
        
        import time
        
        for i, question in enumerate(test_questions):
            start_time = time.time()
            
            try:
                # Get answer
                result = self.generate_answer(question)
                answer = result['answer']
                
                # Calculate retrieval score (based on number of sources found)
                retrieval_score = len(result.get('source_documents', []))
                
                response_time = time.time() - start_time
                
                results['questions'].append(question)
                results['answers'].append(answer)
                results['retrieval_scores'].append(retrieval_score)
                results['response_times'].append(response_time)
                
                print(f"✅ Question {i+1}/{len(test_questions)} processed")
                
            except Exception as e:
                print(f"❌ Error processing question {i+1}: {str(e)}")
                results['answers'].append(f"Error: {str(e)}")
                results['retrieval_scores'].append(0)
                results['response_times'].append(0)
        
        # Calculate summary statistics
        avg_retrieval_score = sum(results['retrieval_scores']) / len(results['retrieval_scores'])
        avg_response_time = sum(results['response_times']) / len(results['response_times'])
        
        results['summary'] = {
            'total_questions': len(test_questions),
            'avg_retrieval_score': avg_retrieval_score,
            'avg_response_time': avg_response_time,
            'llm_provider': self.llm_provider
        }
        
        return results
    
    def export_knowledge_base(self, output_path: str):
        """Export the current knowledge base to a JSON file."""
        if not self.vector_store.vectorstore:
            raise ValueError("No knowledge base to export")
        
        # Get all documents from the vector store
        try:
            # This is a simplified export - in practice, you'd want to export
            # the actual vector embeddings and metadata
            export_data = {
                'timestamp': str(Path().cwd()),
                'embedding_model': 'all-MiniLM-L6-v2',  # Default model
                'chunk_size': self.document_processor.chunk_size,
                'chunk_overlap': self.document_processor.chunk_overlap,
                'llm_provider': self.llm_provider
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✅ Knowledge base exported to {output_path}")
            
        except Exception as e:
            print(f"❌ Error exporting knowledge base: {str(e)}")


def compare_llm_providers():
    """Compare different LLM providers on the same dataset."""
    print("🔄 Comparing LLM Providers")
    print("=" * 50)
    
    # Create sample documents (reuse from demo)
    from demo import create_sample_documents
    doc_paths = create_sample_documents()
    
    # Test questions
    test_questions = [
        "What are the three main steps of RAG?",
        "What are the benefits of using RAG?",
        "How do vector databases work?",
        "What are some popular embedding models?"
    ]
    
    providers = []
    
    # Test OpenAI (if API key available)
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("openai", {}))
    
    # Test Anthropic (if API key available)
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(("anthropic", {}))
    
    # Test local model (if Ollama is available)
    try:
        providers.append(("local", {"model": "llama2"}))
    except:
        print("⚠️  Local LLM (Ollama) not available")
    
    if not providers:
        print("❌ No LLM providers available. Please set API keys or install Ollama.")
        return
    
    results = {}
    
    for provider_name, config in providers:
        print(f"\n🧪 Testing {provider_name.upper()} provider...")
        
        try:
            # Initialize RAG system with specific provider
            rag = AdvancedRAGSystem(llm_provider=provider_name, **config)
            
            # Index documents
            rag.index_documents(doc_paths)
            
            # Evaluate
            evaluation = rag.evaluate_responses(test_questions)
            results[provider_name] = evaluation
            
            print(f"✅ {provider_name.upper()} evaluation completed")
            print(f"   Avg retrieval score: {evaluation['summary']['avg_retrieval_score']:.2f}")
            print(f"   Avg response time: {evaluation['summary']['avg_response_time']:.2f}s")
            
        except Exception as e:
            print(f"❌ Error testing {provider_name}: {str(e)}")
            results[provider_name] = {"error": str(e)}
    
    # Save comparison results
    output_file = "llm_comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Comparison results saved to {output_file}")
    return results


def main():
    """Main function to demonstrate advanced features."""
    print("🚀 Advanced RAG System Demo")
    print("=" * 50)
    
    # Option 1: Compare different LLM providers
    print("\n1. Compare LLM Providers")
    if input("Run LLM provider comparison? (y/n): ").lower() == 'y':
        compare_llm_providers()
    
    # Option 2: Test specific provider
    print("\n2. Test Specific Provider")
    print("Available providers: openai, anthropic, local")
    provider = input("Enter provider name (or press Enter for openai): ").strip() or "openai"
    
    try:
        # Initialize advanced RAG system
        rag = AdvancedRAGSystem(llm_provider=provider)
        
        # Load sample documents
        from demo import create_sample_documents
        doc_paths = create_sample_documents()
        rag.index_documents(doc_paths)
        
        # Interactive Q&A
        print(f"\n💬 Interactive Q&A with {provider.upper()}")
        print("Enter 'quit' to exit")
        
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                try:
                    result = rag.generate_answer(question)
                    print(f"\n🤖 Answer: {result['answer']}")
                    
                    if result.get('source_documents'):
                        print(f"\n📚 Sources ({len(result['source_documents'])}):")
                        for i, source in enumerate(result['source_documents'], 1):
                            source_file = Path(source['metadata'].get('source', 'Unknown')).name
                            print(f"   {i}. {source_file}")
                
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
        
        # Export knowledge base
        if input("\nExport knowledge base? (y/n): ").lower() == 'y':
            rag.export_knowledge_base("knowledge_base_export.json")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n✅ Advanced demo completed!")


if __name__ == "__main__":
    main()
