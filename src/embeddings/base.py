from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract base class for text embedding models"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of documents
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query string
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        pass