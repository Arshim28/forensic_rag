import os
import gc
import tracemalloc
import argparse
from utils import log_memory_usage, logger
from ocr_vector_store import OCRVectorStore
from config import validate_config, DEFAULT_CONFIG

def main():
    tracemalloc.start()
    
    parser = argparse.ArgumentParser(description="OCR Vector Store")
    parser.add_argument("--pdf-dir", help="Directory containing PDF files")
    parser.add_argument("--pdf", help="Single PDF file to process")
    parser.add_argument("--save-dir", default="vector_store", help="Directory to save vector store")
    parser.add_argument("--load-dir", help="Directory to load vector store from")
    parser.add_argument("--query", help="Query to search for")
    parser.add_argument("--index-type", default=DEFAULT_CONFIG["index_type"], 
                      choices=["Flat", "IVF", "IVFPQ", "HNSW"], 
                      help="Type of Faiss index to use")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CONFIG["chunk_size"], 
                      help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CONFIG["chunk_overlap"], 
                      help="Overlap between chunks")
    
    args = parser.parse_args()
    
    logger.warning("PROGRAM START")
    log_memory_usage("program_start")
    
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    vector_store = OCRVectorStore(
        index_type=args.index_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    if args.load_dir:
        logger.info(f"Loading vector store from {args.load_dir}")
        vector_store.load(args.load_dir)
    
    if args.pdf_dir:
        logger.info(f"Processing PDFs in {args.pdf_dir}")
        for filename in os.listdir(args.pdf_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(args.pdf_dir, filename)
                vector_store.add_document(pdf_path)
                
                log_memory_usage(f"after_processing_{filename}")
                gc.collect()
                log_memory_usage(f"after_gc_{filename}")
    
    if args.pdf:
        logger.info(f"Processing PDF: {args.pdf}")
        vector_store.add_document(args.pdf)
    
    if args.save_dir and (args.pdf_dir or args.pdf):
        logger.info(f"Saving vector store to {args.save_dir}")
        vector_store.save(args.save_dir)
    
    if args.query:
        logger.info(f"Searching for: {args.query}")
        results = vector_store.answer_question(args.query)
        
        print("\n--- Search Results ---")
        for i, result in enumerate(results):
            print(f"\n[{i+1}] Score: {result['score']:.4f}")
            print(f"Source: {result['metadata']['source']}, Page: {result['metadata']['page']}")
            print(f"Text: {result['text'][:200]}...")
    
    if not args.query and (args.pdf_dir or args.pdf or args.load_dir):
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            query = input("\nEnter a query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            results = vector_store.answer_question(query)
            
            print("\n--- Search Results ---")
            for i, result in enumerate(results):
                print(f"\n[{i+1}] Score: {result['score']:.4f}")
                print(f"Source: {result['metadata']['source']}, Page: {result['metadata']['page']}")
                print(f"Text: {result['text'][:200]}...")
    
    current, peak = tracemalloc.get_traced_memory()
    logger.warning(f"FINAL MEMORY STATS: Current: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
    tracemalloc.stop()
    
    log_memory_usage("program_end")

if __name__ == "__main__":
    main()