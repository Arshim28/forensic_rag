# OCR-Powered RAG System

A Retrieval-Augmented Generation (RAG) system with OCR capabilities that extracts text from PDF documents, creates embeddings, and enables semantic search.

## Features

- PDF document processing with OCR using Mistral API
- Text chunking with configurable parameters
- Vector embeddings with Google Gemini API
- FAISS-based vector storage with multiple index types (Flat, IVF, IVFPQ, HNSW)
- Semantic search for document retrieval
- Memory-efficient batch processing for large documents
- Progress tracking and memory usage monitoring
- Interactive query mode for document exploration

## Requirements

- Python 3.8+
- Required API Keys:
  - Mistral API Key (for OCR)
  - Google API Key (for Gemini embeddings)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd test_rag
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export MISTRAL_API_KEY="your-mistral-api-key"
```

## Usage

### Process and Search PDF Documents

```bash
# Process a single PDF file and save vector store
python main.py --pdf path/to/document.pdf --save-dir vector_store

# Process multiple PDFs in a directory
python main.py --pdf-dir path/to/pdf/directory --save-dir vector_store

# Load existing vector store and query it
python main.py --load-dir vector_store --query "your search query"

# Process a PDF with custom chunking parameters
python main.py --pdf path/to/document.pdf --chunk-size 5000 --chunk-overlap 200 --index-type HNSW
```

### Interactive Mode

If you process PDFs without providing a query, the system enters interactive mode where you can query the documents repeatedly:

```bash
python main.py --pdf path/to/document.pdf
# Then enter queries when prompted
```

### Supported Index Types

- `Flat`: Basic exact search (default)
- `IVF`: Inverted file index for faster approximate search
- `IVFPQ`: Product quantization for memory-efficient storage
- `HNSW`: Hierarchical Navigable Small World graph for fast approximate search

## Configuration

Default configuration is in `config.py`. Key parameters include:

- `chunk_size`: The size of text chunks for vector storage (default: 10000)
- `chunk_overlap`: Overlap between chunks to avoid splitting contextual information (default: 500)
- `index_type`: Type of FAISS index (default: "Flat")
- `embedding_dimension`: Dimension of the embeddings (default: 3072)
- `max_tokens`: Maximum tokens allowed in a single embedding request (default: 8000)

## Project Structure

- `main.py`: Entry point for the application
- `ocr_processor.py`: Handles OCR processing using Mistral API
- `embedding.py`: Manages embedding generation with Google Gemini
- `document_processor.py`: Processes documents and handles chunking
- `vector_store.py`: Implements FAISS vector storage and retrieval
- `ocr_vector_store.py`: Combines OCR and vector storage for document search
- `text_chunk.py`: Class to represent text chunks with metadata
- `utils.py`: Utility functions for logging and memory management
- `config.py`: Configuration settings

## Performance Considerations

- For large PDFs, the system uses batch processing to optimize memory usage
- Memory usage monitoring helps prevent out-of-memory errors
- Different index types can be selected based on the speed/accuracy tradeoff needed

## License

[License information]

## Acknowledgements

- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Mistral AI](https://mistral.ai/) for OCR capabilities
- [Google Gemini](https://ai.google.dev/) for embeddings