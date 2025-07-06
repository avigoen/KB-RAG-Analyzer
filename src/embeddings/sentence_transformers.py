import logging
from typing import List, Optional

from .base import EmbeddingModel

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model using Sentence Transformers"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize the Sentence Transformer embedding model
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on (None for auto-detection)
            normalize_embeddings: Whether to normalize embeddings (recommended for cosine similarity)
            batch_size: Batch size for processing multiple documents
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence Transformers is not installed. Install with 'pip install sentence-transformers'")
        
        self.model_name = model_name
        self.normalize = normalize_embeddings
        self.batch_size = batch_size
        
        # Load the model
        self.model = SentenceTransformer(model_name, device=device)
        
        logger.info(f"Initialized SentenceTransformerEmbedding with model '{model_name}'")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of documents"""
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # Convert numpy array to list of lists
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query"""
        embedding = self.model.encode(
            text, 
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
