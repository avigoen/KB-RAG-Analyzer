"""
Complete RAG pipeline implementation that ties together all components:
1. Document parsing
2. Document chunking
3. Document embedding
4. Vector storage
5. Query processing and retrieval
6. LLM generation

This module orchestrates the entire RAG workflow.
"""
from typing import Any, Dict, List, Optional

import json
import logging
import os
import time
from pathlib import Path

# Import our modules
from .embeddings import EmbeddingModelFactory, EmbeddingModel
from .llm import LLMFactory, LLM
from .parsers.chunking import DocumentChunker
from .parsers.pipeline import PositionalDocumentParsingPipeline
from .vector_store import VectorStoreFactory, VectorStore


class RAGPipeline:
    """
    Complete RAG pipeline that integrates document parsing, chunking,
    embedding, vector storage, and LLM generation.
    """
    
    def __init__(
        self,
        embedding_model_type: str = "sentence-transformer",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        vector_store_type: str = "chroma",
        vector_store_params: Dict[str, Any] = None,
        llm_type: str = "huggingface",
        llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        chunking_strategy: str = "semantic",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        data_dir: str = "./data",
        log_level: str = "INFO"
    ):
        """
        Initialize the RAG pipeline with all components
        
        Args:
            embedding_model_type: Type of embedding model (sentence-transformer, huggingface, openai-compatible)
            embedding_model_name: Name of the embedding model
            vector_store_type: Type of vector store (chroma, milvus)
            vector_store_params: Additional parameters for vector store initialization
            llm_type: Type of LLM (huggingface, llamacpp, openai-compatible)
            llm_model_name: Name of the LLM model
            chunking_strategy: Strategy for document chunking (fixed, semantic, hierarchical)
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            data_dir: Directory for storing data
            log_level: Logging level
        """
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create data directories
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.parsed_dir = self.data_dir / "parsed"
        self.chunks_dir = self.data_dir / "chunks"
        self.vectors_dir = self.data_dir / "vectors"
        
        for directory in [self.data_dir, self.raw_dir, self.parsed_dir, self.chunks_dir, self.vectors_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Initialize document parsing pipeline
        self.parser = PositionalDocumentParsingPipeline()
        
        # Initialize document chunker
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy,
            preserve_metadata=True
        )
        
        # Initialize embedding model
        self.embedding_model = EmbeddingModelFactory.create_embedding_model(
            model_type=embedding_model_type,
            model_name=embedding_model_name
        )
        
        # Initialize vector store with appropriate parameters
        vector_store_params = vector_store_params or {}
        if vector_store_type == "chroma" and "persist_directory" not in vector_store_params:
            vector_store_params["persist_directory"] = str(self.vectors_dir)
            
        self.vector_store = VectorStoreFactory.create_vector_store(
            store_type=vector_store_type,
            embedding_dimension=self.embedding_model.embedding_dimension,
            **vector_store_params
        )
        
        # Initialize LLM
        self.llm = LLMFactory.create_llm(
            llm_type=llm_type,
            model_name=llm_model_name
        )
        
        self.logger.info("RAG pipeline initialized successfully")
    
    def process_document(self, file_path: str, save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Process a single document through the entire pipeline up to vector storage
        
        Args:
            file_path: Path to the document file
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        stats = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size_bytes": os.path.getsize(file_path),
            "stages": {}
        }
        
        # 1. Parse the document
        self.logger.info(f"Parsing document: {file_path}")
        parse_start = time.time()
        parsed_doc = self.parser.parse_document(file_path)
        parse_time = time.time() - parse_start
        
        stats["stages"]["parsing"] = {
            "time_seconds": parse_time,
            "elements_extracted": len(parsed_doc.structure.elements),
            "pages": len(parsed_doc.structure.page_dimensions)
        }
        
        # Save parsed document if requested
        if save_intermediate:
            parsed_file = self.parsed_dir / f"{Path(file_path).stem}.json"
            with open(parsed_file, "w") as f:
                # Convert to dict for JSON serialization
                doc_dict = {
                    "metadata": parsed_doc.metadata,
                    "file_path": parsed_doc.file_path,
                    "file_type": parsed_doc.file_type,
                    "parser_used": parsed_doc.parser_used,
                    "content": parsed_doc.content,
                    "elements_count": len(parsed_doc.structure.elements),
                    "errors": parsed_doc.errors
                }
                json.dump(doc_dict, f, indent=2)
        
        # 2. Chunk the document
        self.logger.info(f"Chunking document: {file_path}")
        chunk_start = time.time()
        chunks = self.chunker.chunk_document(parsed_doc)
        chunk_time = time.time() - chunk_start
        
        stats["stages"]["chunking"] = {
            "time_seconds": chunk_time,
            "chunks_created": len(chunks),
            "strategy": self.chunker.strategy
        }
        
        # Save chunks if requested
        if save_intermediate:
            chunks_file = self.chunks_dir / f"{Path(file_path).stem}_chunks.json"
            with open(chunks_file, "w") as f:
                json.dump(chunks, f, indent=2)
        
        # 3. Embed the chunks
        self.logger.info(f"Embedding {len(chunks)} chunks from document: {file_path}")
        embed_start = time.time()
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        embed_time = time.time() - embed_start
        
        stats["stages"]["embedding"] = {
            "time_seconds": embed_time,
            "chunks_embedded": len(embeddings),
            "model_used": self.embedding_model.__class__.__name__,
            "embedding_dimensions": self.embedding_model.embedding_dimension
        }
        
        # 4. Store vectors
        self.logger.info(f"Storing vectors in database: {file_path}")
        store_start = time.time()
        ids = self.vector_store.add_documents(chunks, embeddings)
        store_time = time.time() - store_start
        
        stats["stages"]["vector_storage"] = {
            "time_seconds": store_time,
            "vectors_stored": len(ids),
            "store_type": self.vector_store.__class__.__name__
        }
        
        stats["total_time_seconds"] = time.time() - start_time
        self.logger.info(f"Document processing completed: {file_path} in {stats['total_time_seconds']:.2f} seconds")
        
        return stats
    
    def process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List of processing statistics dictionaries
        """
        stats = []
        for file_path in file_paths:
            try:
                doc_stats = self.process_document(file_path)
                stats.append(doc_stats)
            except Exception as e:
                self.logger.error(f"Error processing document {file_path}: {str(e)}")
                stats.append({
                    "file_path": file_path,
                    "error": str(e),
                    "status": "failed"
                })
        
        return stats
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Execute a query against the RAG system
        
        Args:
            query: User query string
            top_k: Number of relevant chunks to retrieve
            filter_criteria: Optional filter criteria for vector search
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens for LLM response
            
        Returns:
            Dictionary with query results including answer and sources
        """
        # 1. Generate embedding for query
        self.logger.info(f"Processing query: {query}")
        query_embedding = self.embedding_model.embed_query(query)
        
        # 2. Search vector database for similar chunks
        search_results = self.vector_store.query_by_embedding(
            embedding=query_embedding,
            k=top_k,
            filter_criteria=filter_criteria
        )
        
        # 3. Prepare context from retrieved chunks
        contexts = []
        sources = []
        
        for result in search_results:
            # Add text content as context
            contexts.append(result["text"])
            
            # Capture source information
            source_info = {
                "score": result.get("score", 0),
                "chunk_id": result.get("chunk_id", "")
            }
            
            # Add metadata if available
            if "metadata" in result:
                source_info["source"] = result["metadata"].get("source", "")
                source_info["page_range"] = result["metadata"].get("page_range", [])
            
            sources.append(source_info)
        
        # 4. Generate answer with LLM
        answer = self.llm.generate_with_context(
            question=query,
            context=contexts,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 5. Return query results
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(contexts),
            "timestamp": time.time()
        }
    
    def load_document_from_path(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load all documents from a directory path
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processing statistics
        """
        path = Path(directory_path)
        file_paths = []
        
        # Find all document files
        for file_path in path.glob("**/*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                file_paths.append(str(file_path))
        
        self.logger.info(f"Found {len(file_paths)} documents in {directory_path}")
        
        # Process all documents
        return self.process_documents(file_paths)
    
    def clear_vector_store(self) -> None:
        """
        Clear the vector store, removing all documents
        """
        # This is a placeholder - actual implementation depends on vector store type
        self.logger.warning("Vector store clearing is not implemented for this vector store type")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "document_count": self.vector_store.count(),
            "embedding_model": {
                "type": self.embedding_model.__class__.__name__,
                "dimensions": self.embedding_model.embedding_dimension
            },
            "vector_store": {
                "type": self.vector_store.__class__.__name__
            },
            "llm": {
                "type": self.llm.__class__.__name__
            },
            "chunking": {
                "strategy": self.chunker.strategy,
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap
            }
        }
        
        return stats 