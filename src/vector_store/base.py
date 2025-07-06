from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """Abstract base class for vector database interactions"""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """
        Add documents with their embeddings to the vector store
        
        Args:
            documents: List of document chunks with metadata
            embeddings: List of embeddings as vectors
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def get_documents_by_id(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by their IDs
        
        Args:
            ids: List of document IDs
            
        Returns:
            List of document dictionaries
        """
        pass
    
    @abstractmethod
    def query_by_embedding(self, 
                          embedding: List[float], 
                          k: int = 5, 
                          filter_criteria: Optional[Dict[str, Any]] = None
                          ) -> List[Dict[str, Any]]:
        """
        Query the vector store using an embedding vector
        
        Args:
            embedding: The query embedding vector
            k: Number of similar documents to retrieve
            filter_criteria: Optional filters to apply
            
        Returns:
            List of document dictionaries with similarity scores
        """
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs
        
        Args:
            ids: List of document IDs to delete
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Count the number of documents in the store
        
        Returns:
            Document count
        """
        pass