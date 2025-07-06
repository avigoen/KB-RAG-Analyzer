# Knowledgebase RAG Analyzer

A comprehensive Retrieval-Augmented Generation (RAG) pipeline for analyzing and querying documents in enterprise environments. This system can parse, process, and query various document formats, starting with PDFs and extending to other formats.

## Features

- **Document Parsing**: Extract text, tables, and structure from multiple document formats
- **Intelligent Chunking**: Split documents with multiple strategies (fixed, semantic, hierarchical)
- **Embedding Generation**: Transform text into vector representations using various embedding models
- **Vector Storage**: Store and search document embeddings with Chroma or Milvus
- **LLM Integration**: Connect to various LLM backends for question answering
- **Modular Architecture**: Flexible and extensible components

## Architecture

The system follows the standard RAG pipeline architecture:

1. **Document Ingestion**: Parse documents and extract content with spatial awareness
2. **Preprocessing & Chunking**: Clean text and split into optimal chunks
3. **Embedding Generation**: Convert chunks to vector representations
4. **Vector Storage**: Store embeddings in vector databases
5. **Query Processing**: Retrieve relevant context based on user queries
6. **LLM Generation**: Generate answers by augmenting LLMs with retrieved context

## Installation

### Prerequisites

- Python 3.8+
- Sufficient storage space for documents and embeddings
- GPU(s) recommended for better performance (optional)

### Setup

1. Clone this repository:

```bash
git clone https://github.com/yourusername/knowledgebase-rag-analyzer.git
cd knowledgebase-rag-analyzer
```

2. Install required packages:

```bash
pip install .
```

3. Install optional dependencies for specific vector stores:

```bash
# For Milvus support
pip install pymilvus>=2.3.0
```

## Usage

### Document Processing

Process PDF documents and add them to the vector store:

```bash
python src/demo.py --action process --pdf_file path/to/your/document.pdf
```

Process all PDFs in a directory:

```bash
python src/demo.py --action process --docs_dir path/to/your/documents
```

### Query the System

Ask questions about the processed documents:

```bash
python src/demo.py --action query --query "What is the main topic discussed in the documents?"
```

### Configuration Options

The system can be configured with various options:

```bash
# Use different embedding model
python src/demo.py --embedding_model huggingface --action process --pdf_file document.pdf

# Use different chunking strategy
python src/demo.py --chunk_strategy hierarchical --chunk_size 1000 --action process --pdf_file document.pdf

# Configure the LLM backend
python src/demo.py --llm_type llamacpp --llm_model llama2 --action query --query "Summarize the key points"
```

### Show Statistics

View statistics about the RAG system:

```bash
python src/demo.py --action stats
```

## Supported Document Types

Currently, the system supports the following document formats:

- PDF documents (text, tables, images with OCR)
- HTML files (planned extension)
- Microsoft Word documents (planned extension)
- Excel spreadsheets (planned extension)
- Plain text files (planned extension)
- Image files with text (planned extension)

## Embedding Models

The following embedding models are supported:

- Sentence Transformers (default: "all-MiniLM-L6-v2")
- Hugging Face Transformers models
- OpenAI-compatible API endpoints

## Vector Stores

Supported vector database backends:

- ChromaDB (default)
- Milvus (optional)

## LLM Backends

The system can integrate with various LLM backends:

- Hugging Face Transformers
- llama.cpp server
- OpenAI-compatible APIs

## Advanced Configuration

For advanced use cases, you can directly use the RAGPipeline class in your code:

```python
from src.rag_pipeline import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline(
    embedding_model_type="sentence-transformer",
    embedding_model_name="all-MiniLM-L6-v2",
    vector_store_type="chroma",
    llm_type="huggingface",
    llm_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    chunking_strategy="semantic",
    chunk_size=500,
    chunk_overlap=50,
    data_dir="./data"
)

# Process a document
pipeline.process_document("path/to/document.pdf")

# Query the system
result = pipeline.query(
    query="What is the main topic discussed in the document?",
    top_k=5
)

print(result["answer"])
```

## Project Structure

- `src/parsers/`: Document parsing modules
- `src/embeddings.py`: Embedding model implementations
- `src/vector_store.py`: Vector database interfaces
- `src/llm.py`: LLM backend interfaces
- `src/rag_pipeline.py`: Main RAG pipeline implementation
- `src/demo.py`: Demo script for CLI usage
- `notebooks/`: Example notebooks

## License

This project is licensed under the terms of the license included with this repository.

## Acknowledgments

This project was inspired by research on production-scale RAG systems and designed to provide a modular, extensible framework for document analysis and retrieval.
