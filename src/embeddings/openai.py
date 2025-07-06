import logging
from typing import List, Optional

from .base import EmbeddingModel


logger = logging.getLogger(__name__)

class OpenAICompatibleEmbedding(EmbeddingModel):
    """
    Embedding model using an OpenAI-compatible API
    (for open-source models with OpenAI-compatible endpoints like Ollama)
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-ada-002",
        embedding_dimension: int = 1536,
        batch_size: int = 32
    ):
        """
        Initialize an embedding model using an OpenAI-compatible API
        
        Args:
            api_url: URL of the API endpoint
            api_key: Optional API key
            model_name: Name of the embedding model
            embedding_dimension: Dimension of the embedding vectors
            batch_size: Batch size for processing multiple documents
        """
        # Optional import to avoid dependency issues
        import requests
        self.requests = requests
        
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self._embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        
        # Headers for API requests
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        logger.info(f"Initialized OpenAICompatibleEmbedding with endpoint '{api_url}'")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of documents"""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a single batch of texts"""
        payload = {
            "input": texts,
            "model": self.model_name
        }
        
        response = self.requests.post(
            f"{self.api_url}/embeddings",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query"""
        return self._embed_batch([text])[0]
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self._embedding_dimension