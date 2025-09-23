# RAG System

A complete Retrieval-Augmented Generation (RAG) system implementation that allows you to create your own knowledge base and query it using Large Language Models.

## Features

- **Document Processing**: Support for text files and PDFs
- **Vector Storage**: ChromaDB for efficient similarity search
- **Multiple LLM Support**: OpenAI, Anthropic, and local models
- **Web Interface**: Streamlit-based GUI for easy interaction
- **Evaluation Tools**: Built-in performance evaluation and comparison

## Quick Start

### 1. Installation

```bash
# Clone or download the files to your directory
cd How_RAG_works

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy the example environment file
copy .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Demo

```bash
# Basic demo with sample documents
python demo.py

# Web interface
streamlit run streamlit_app.py

# Advanced features comparison
python advanced_rag.py
```

## System Architecture

The RAG system follows the three-step process outlined in your notes:

### 1. Indexing
- **Document Loading**: Supports `.txt`, `.pdf`, and other text formats
- **Text Chunking**: Splits documents into manageable pieces
- **Embedding**: Converts text to vector representations
- **Storage**: Stores embeddings in ChromaDB vector database

### 2. Retrieval
- **Query Processing**: Converts user questions to embeddings
- **Similarity Search**: Finds most relevant document chunks
- **Context Selection**: Returns top-k most relevant pieces

### 3. Generation
- **Context Injection**: Combines retrieved documents with user query
- **LLM Processing**: Generates answers using the augmented prompt
- **Response Formatting**: Returns structured answers with sources

## Usage Examples

### Basic Usage

```python
from rag_system import RAGSystem

# Initialize the system
rag = RAGSystem()

# Index your documents
documents = ["path/to/doc1.txt", "path/to/doc2.pdf"]
rag.index_documents(documents)

# Ask questions
answer = rag.ask("What is the main topic of the documents?")
print(answer)
```

### Advanced Usage

```python
from advanced_rag import AdvancedRAGSystem

# Use different LLM providers
rag_openai = AdvancedRAGSystem(llm_provider="openai")
rag_anthropic = AdvancedRAGSystem(llm_provider="anthropic")
rag_local = AdvancedRAGSystem(llm_provider="local", model="llama2")

# Evaluate performance
test_questions = ["What is RAG?", "How does it work?"]
results = rag.evaluate_responses(test_questions)
```

### Web Interface

The Streamlit interface provides:
- Document upload and indexing
- Real-time question answering
- Source document viewing
- Chat history

## Configuration Options

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for GPT models
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TEMPERATURE`: LLM temperature (default: 0.7)

### Customization

```python
# Custom chunk sizes
rag = RAGSystem()
rag.document_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)

# Custom embedding model
rag.vector_store = VectorStore(embedding_model="sentence-transformers/all-mpnet-base-v2")

# Custom prompt template
from langchain.prompts import PromptTemplate
custom_prompt = PromptTemplate(
    template="Based on the context: {context}\nAnswer: {question}",
    input_variables=["context", "question"]
)
rag.prompt_template = custom_prompt
```

## File Structure

```
How_RAG_works/
├── rag_system.py          # Core RAG implementation
├── demo.py                # Basic demonstration
├── streamlit_app.py       # Web interface
├── advanced_rag.py        # Advanced features
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
├── README.md             # This file
└── sample_documents/     # Demo documents (created by demo.py)
    ├── rag_overview.txt
    ├── vector_databases.txt
    └── llm_integration.txt
```

## Key Benefits (from your notes)

✅ **Reduced hallucinations**: Grounds responses in actual documents  
✅ **Access to current data**: Uses your specific knowledge base  
✅ **Increased trust and transparency**: Shows source documents  
✅ **Lower costs**: No need for fine-tuning large models  
✅ **Enhanced customization**: Easy to add domain-specific documents  

## Comparison with Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Cost** | Lower | Higher |
| **Data Updates** | Real-time | Requires retraining |
| **Transparency** | Shows sources | Black box |
| **Domain Adaptation** | Easy | Complex |
| **Hallucinations** | Reduced | May increase |

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing packages with `pip install -r requirements.txt`
2. **API Key Errors**: Check your `.env` file and API key validity
3. **Memory Issues**: Reduce chunk size or use smaller embedding models
4. **Slow Performance**: Consider using cloud vector databases for large datasets

### Performance Optimization

- Use appropriate chunk sizes (500-1500 characters)
- Choose embedding models based on your language/domain
- Implement caching for repeated queries
- Use async processing for large document sets

## Real-world Applications

Based on your notes, here are some practical applications:

1. **Academic Research**: RAG for academic papers (like your reference to the survey paper)
2. **Customer Support**: Company knowledge base integration
3. **Legal Research**: Case law and regulation querying
4. **Medical Information**: Clinical guidelines and research papers
5. **Code Documentation**: Software documentation search

## Integration with Agentic AI

As mentioned in your notes, RAG can be integrated with agentic AI systems:

```python
# Example: RAG as a tool for an AI agent
class AIAgent:
    def __init__(self):
        self.rag_tool = RAGSystem()
        self.tools = [self.rag_tool, other_tools...]
    
    def answer_query(self, query):
        # Agent decides whether to use RAG or other tools
        if self.needs_knowledge_lookup(query):
            return self.rag_tool.ask(query)
        else:
            return self.other_processing(query)
```

## Model Context Protocol (MCP) Integration

Your notes mention MCP - here's how RAG fits:

```python
# RAG as an MCP server component
class RAGMCPServer:
    def __init__(self):
        self.rag = RAGSystem()
    
    def handle_rag_request(self, query, context):
        # Process RAG request in MCP environment
        return self.rag.generate_answer(query)
```

## Next Steps

1. **Try the basic demo**: Run `python demo.py` to see it in action
2. **Add your documents**: Replace sample documents with your own files
3. **Customize the system**: Adjust chunk sizes, embedding models, and prompts
4. **Scale up**: Consider cloud deployment for production use
5. **Integrate**: Connect with your existing applications

## Contributing

Feel free to extend this system with:
- Additional LLM providers
- Different vector databases
- Advanced retrieval techniques
- Custom evaluation metrics
- Domain-specific optimizations

---

This RAG system implements all the concepts from your research notes and provides a foundation for building more sophisticated knowledge-based applications!
