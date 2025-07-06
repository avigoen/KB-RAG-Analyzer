import os
import logging
from typing import Any, Dict, List, Optional
import uuid

from .base import VectorStore

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class ChromaVectorStore(VectorStore):
    """Chroma vector database implementation"""
    
    def __init__(
        self, 
        collection_name: str = "document_collection",
        persist_directory: Optional[str] = None,
        embedding_dimension: int = 768
    ):
        """
        Initialize Chroma vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data (None for in-memory)
            embedding_dimension: Dimension of the embedding vectors
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with 'pip install chromadb'")
        
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Initialize client
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Initialized ChromaVectorStore with collection '{collection_name}'")
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """Add documents with embeddings to Chroma"""
        if len(documents) != len(embeddings):
            raise ValueError(f"Number of documents ({len(documents)}) and embeddings ({len(embeddings)}) must match")
        
        # Prepare document data for Chroma
        ids = [doc.get('chunk_id', str(uuid.uuid4())) for doc in documents]
        texts = [doc['text'] for doc in documents]
        metadatas = [
            {k: v for k, v in doc.items() if k != 'text' and k != 'chunk_id'}
            for doc in documents
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        return ids
    
    def get_documents_by_id(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Get documents by their IDs from Chroma"""
        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"]
        )
        
        documents = []
        for i, doc_id in enumerate(results['ids']):
            doc = {
                'chunk_id': doc_id,
                'text': results['documents'][i]
            }
            # Add metadata
            if results['metadatas'][i]:
                doc.update(results['metadatas'][i])
            documents.append(doc)
        
        return documents
    
    def query_by_embedding(self, 
                          embedding: List[float], 
                          k: int = 5, 
                          filter_criteria: Optional[Dict[str, Any]] = None
                          ) -> List[Dict[str, Any]]:
        """Query Chroma using an embedding vector"""
        # Convert filter criteria to Chroma format if provided
        where_document = filter_criteria if filter_criteria else None
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where_document,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        for i, doc_id in enumerate(results['ids'][0]):
            doc = {
                'chunk_id': doc_id,
                'text': results['documents'][0][i],
                'score': 1.0 - float(results['distances'][0][i])  # Convert distance to similarity
            }
            # Add metadata
            if results['metadatas'][0][i]:
                doc.update(results['metadatas'][0][i])
            documents.append(doc)
        
        return documents
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by their IDs from Chroma"""
        self.collection.delete(ids=ids)
    
    def count(self) -> int:
        """Count documents in the Chroma collection"""
        return self.collection.count()