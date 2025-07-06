"""
Demo script to showcase the RAG pipeline functionality with PDF documents.
This script demonstrates how to:
1. Initialize the RAG pipeline
2. Process PDF documents
3. Query the system with questions
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import time
import json

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline


def setup_argparser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description="RAG Pipeline Demo for PDF Documents")
    
    # Action options
    parser.add_argument("--action", type=str, choices=["process", "query", "stats"], 
                      default="process", help="Action to perform: process documents, query the system, or show statistics")
    
    # Document processing options
    parser.add_argument("--docs_dir", type=str, default="./data/raw", 
                      help="Directory containing PDF documents to process")
    parser.add_argument("--pdf_file", type=str, 
                      help="Path to a specific PDF file to process")
    
    # Query options
    parser.add_argument("--query", type=str, 
                      help="Question to ask the RAG system")
    parser.add_argument("--top_k", type=int, default=5,
                      help="Number of relevant chunks to retrieve")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="Temperature parameter for LLM generation")
    
    # Configuration options
    parser.add_argument("--data_dir", type=str, default="./data",
                      help="Directory for storing pipeline data")
    parser.add_argument("--chunk_size", type=int, default=500,
                      help="Size of document chunks")
    parser.add_argument("--chunk_strategy", type=str, default="semantic", 
                      choices=["fixed", "semantic", "hierarchical"],
                      help="Chunking strategy")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformer",
                      choices=["sentence-transformer", "huggingface", "openai-compatible"],
                      help="Type of embedding model")
    parser.add_argument("--vector_store", type=str, default="chroma",
                      choices=["chroma", "milvus"],
                      help="Type of vector store")
    parser.add_argument("--llm_type", type=str, default="huggingface",
                      choices=["huggingface", "llamacpp", "openai-compatible"],
                      help="Type of LLM")
    parser.add_argument("--llm_model", type=str,
                      help="Name of the LLM model")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    
    return parser


def process_documents(pipeline, args):
    """Process documents with the RAG pipeline"""
    if args.pdf_file:
        # Process a single PDF file
        if not os.path.exists(args.pdf_file):
            print(f"Error: PDF file not found: {args.pdf_file}")
            sys.exit(1)
            
        print(f"Processing PDF file: {args.pdf_file}")
        stats = pipeline.process_document(args.pdf_file)
        
        # Print stats
        print(f"\nDocument processing statistics:")
        print(f"  - File: {stats['file_name']}")
        print(f"  - Size: {stats['file_size_bytes'] / 1024:.2f} KB")
        print(f"  - Elements extracted: {stats['stages']['parsing']['elements_extracted']}")
        print(f"  - Pages: {stats['stages']['parsing']['pages']}")
        print(f"  - Chunks created: {stats['stages']['chunking']['chunks_created']}")
        print(f"  - Total processing time: {stats['total_time_seconds']:.2f} seconds")
        
    else:
        # Process all PDFs in directory
        if not os.path.exists(args.docs_dir):
            print(f"Error: Documents directory not found: {args.docs_dir}")
            sys.exit(1)
            
        # Find PDF files
        pdf_files = []
        for root, _, files in os.walk(args.docs_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            print(f"No PDF files found in directory: {args.docs_dir}")
            sys.exit(1)
            
        print(f"Found {len(pdf_files)} PDF files in {args.docs_dir}")
        
        # Process all PDF files
        start_time = time.time()
        stats = pipeline.process_documents(pdf_files)
        total_time = time.time() - start_time
        
        # Print summary
        successful = len([s for s in stats if "error" not in s])
        failed = len(stats) - successful
        
        print(f"\nDocument processing summary:")
        print(f"  - Total documents: {len(stats)}")
        print(f"  - Successfully processed: {successful}")
        print(f"  - Failed: {failed}")
        print(f"  - Total processing time: {total_time:.2f} seconds")
        
        if failed > 0:
            print("\nFailed documents:")
            for stat in stats:
                if "error" in stat:
                    print(f"  - {stat['file_path']}: {stat['error']}")


def query_system(pipeline, args):
    """Query the RAG system"""
    if not args.query:
        print("Error: Please provide a query with --query")
        sys.exit(1)
        
    print(f"Querying RAG system: \"{args.query}\"")
    print(f"Retrieving {args.top_k} most relevant chunks...")
    
    # Execute query
    result = pipeline.query(
        query=args.query,
        top_k=args.top_k,
        temperature=args.temperature
    )
    
    # Print result
    print("\nAnswer:")
    print(f"{result['answer']}")
    
    # Print sources
    print("\nSources:")
    for i, source in enumerate(result['sources']):
        print(f"  {i+1}. {source.get('source', 'Unknown')} "
              f"(page {source.get('page_range', [0, 0])[0]}-{source.get('page_range', [0, 0])[1]}), "
              f"score: {source.get('score', 0):.4f}")


def show_statistics(pipeline):
    """Show statistics about the RAG system"""
    stats = pipeline.get_statistics()
    
    print("\nRAG System Statistics:")
    print(f"  - Documents/chunks in vector store: {stats['document_count']}")
    print(f"  - Embedding model: {stats['embedding_model']['type']} "
          f"({stats['embedding_model']['dimensions']} dimensions)")
    print(f"  - Vector store: {stats['vector_store']['type']}")
    print(f"  - LLM: {stats['llm']['type']}")
    print(f"  - Chunking strategy: {stats['chunking']['strategy']} "
          f"(size: {stats['chunking']['chunk_size']}, "
          f"overlap: {stats['chunking']['chunk_overlap']})")


def main():
    """Main function to run the demo"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the pipeline
    print("Initializing RAG pipeline...")
    
    # Prepare vector store params
    vector_store_params = {
        "persist_directory": os.path.join(args.data_dir, "vectors")
    }
    
    pipeline = RAGPipeline(
        embedding_model_type=args.embedding_model,
        vector_store_type=args.vector_store,
        vector_store_params=vector_store_params,
        llm_type=args.llm_type,
        llm_model_name=args.llm_model,
        chunking_strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        data_dir=args.data_dir,
        log_level=args.log_level
    )
    
    # Perform the requested action
    if args.action == "process":
        process_documents(pipeline, args)
    elif args.action == "query":
        query_system(pipeline, args)
    elif args.action == "stats":
        show_statistics(pipeline)
    else:
        print(f"Error: Unknown action: {args.action}")
        sys.exit(1)
        
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main() 