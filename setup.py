"""
Setup script for the RAG system
"""

import os
import sys
from pathlib import Path
import subprocess


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing packages")
        return False


def setup_environment():
    """Setup environment variables."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📄 Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your API keys")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env template found")


def test_basic_functionality():
    """Test basic RAG system functionality."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test imports
        from rag_system import RAGSystem, DocumentProcessor, VectorStore
        print("✅ RAG system imports successful")
        
        # Test document processor
        processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        # Test vector store (without documents)
        vector_store = VectorStore()
        print("✅ Vector store initialized")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False


def main():
    """Main setup function."""
    print("🚀 RAG System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Setup environment
    setup_environment()
    
    # Test functionality
    if test_basic_functionality():
        print("\n" + "=" * 50)
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your OpenAI API key")
        print("2. Run: python demo.py")
        print("3. Or run: streamlit run streamlit_app.py")
        print("4. Read README.md for detailed usage instructions")
    else:
        print("\n" + "=" * 50)
        print("⚠️  Setup completed with warnings")
        print("Some components may not work until dependencies are resolved")


if __name__ == "__main__":
    main()
