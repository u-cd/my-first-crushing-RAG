import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
from typing import List

# Add the parent directory to the path to import rag_system
sys.path.append(str(Path(__file__).parent))

from rag_system import RAGSystem

# Page config
st.set_page_config(
    page_title="RAG System Interface",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'indexed_files' not in st.session_state:
    st.session_state.indexed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def initialize_rag_system(api_key: str):
    """Initialize the RAG system with API key."""
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.rag_system = RAGSystem(openai_api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False


def process_uploaded_files(uploaded_files) -> List[str]:
    """Process uploaded files and save them temporarily."""
    temp_paths = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_paths.append(tmp_file.name)
    
    return temp_paths


def main():
    st.title("🤖 RAG System Interface")
    st.markdown("Upload documents, index them, and ask questions!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable the generation component"
        )
        
        if api_key and not st.session_state.rag_system:
            if st.button("Initialize RAG System"):
                if initialize_rag_system(api_key):
                    st.success("RAG system initialized successfully!")
                    st.rerun()
        
        # System status
        st.header("📊 System Status")
        if st.session_state.rag_system:
            st.success("✅ RAG System Ready")
        else:
            st.warning("⚠️ RAG System Not Initialized")
        
        if st.session_state.indexed_files:
            st.info(f"📄 {len(st.session_state.indexed_files)} files indexed")
        else:
            st.info("📄 No files indexed yet")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Document Upload & Indexing")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose files to index",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'md'],
            help="Upload text files or PDFs to be indexed by the RAG system"
        )
        
        if uploaded_files and st.session_state.rag_system:
            if st.button("Index Documents", type="primary"):
                with st.spinner("Processing and indexing documents..."):
                    try:
                        # Process uploaded files
                        temp_paths = process_uploaded_files(uploaded_files)
                        
                        # Index documents
                        st.session_state.rag_system.index_documents(temp_paths)
                        st.session_state.indexed_files = [f.name for f in uploaded_files]
                        
                        # Clean up temporary files
                        for path in temp_paths:
                            try:
                                os.unlink(path)
                            except:
                                pass
                        
                        st.success(f"Successfully indexed {len(uploaded_files)} documents!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error indexing documents: {str(e)}")
        
        # Show indexed files
        if st.session_state.indexed_files:
            st.subheader("📋 Indexed Files")
            for i, filename in enumerate(st.session_state.indexed_files, 1):
                st.write(f"{i}. {filename}")
    
    with col2:
        st.header("💬 Chat Interface")
        
        if not st.session_state.rag_system:
            st.warning("Please initialize the RAG system first by entering your OpenAI API key in the sidebar.")
        elif not st.session_state.indexed_files:
            st.warning("Please upload and index some documents first.")
        else:
            # Chat interface
            with st.container():
                # Display chat history
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    with st.chat_message("user"):
                        st.write(question)
                    with st.chat_message("assistant"):
                        st.write(answer)
                
                # Question input
                question = st.chat_input("Ask a question about your documents...")
                
                if question:
                    # Add user question to chat
                    with st.chat_message("user"):
                        st.write(question)
                    
                    # Generate answer
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                result = st.session_state.rag_system.generate_answer(question)
                                answer = result['answer']
                                st.write(answer)
                                
                                # Show sources if available
                                if result.get('source_documents'):
                                    with st.expander("📚 View Sources"):
                                        for i, source in enumerate(result['source_documents'], 1):
                                            st.write(f"**Source {i}:**")
                                            st.write(source['content'])
                                            if source.get('metadata', {}).get('source'):
                                                source_file = Path(source['metadata']['source']).name
                                                st.write(f"*From: {source_file}*")
                                            st.write("---")
                                
                                # Add to chat history
                                st.session_state.chat_history.append((question, answer))
                                
                            except Exception as e:
                                error_msg = f"Error generating answer: {str(e)}"
                                st.error(error_msg)
                                st.session_state.chat_history.append((question, error_msg))
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips:**\n"
        "- Upload multiple documents for better context\n"
        "- Ask specific questions about your documents\n"
        "- Check the sources to verify information\n"
        "- The system works best with text-based documents"
    )


if __name__ == "__main__":
    main()
