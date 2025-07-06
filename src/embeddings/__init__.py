import logging
from typing import Optional

from .base import EmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingModelFactory:
    """Factory for creating embedding models"""
    
    @staticmethod
    def create_embedding_model(
        model_type: str,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> EmbeddingModel:
        """
        Create an embedding model of the specified type
        
        Args:
            model_type: Type of embedding model ("sentence-transformer", "huggingface", "openai-compatible")
            model_name: Name of the model to use (depends on model_type)
            device: Device to run the model on (None for auto-detection)
            **kwargs: Additional arguments for the specific model type
            
        Returns:
            EmbeddingModel instance
        """
        if model_type == "sentence-transformer":
            from .sentence_transformers import SentenceTransformerEmbedding
            return SentenceTransformerEmbedding(
                model_name=model_name or "all-MiniLM-L6-v2",
                device=device,
                **kwargs
            )
        elif model_type == "huggingface":
            from .huggingface import HuggingFaceEmbedding
            return HuggingFaceEmbedding(
                model_name=model_name or "intfloat/e5-base-v2",
                device=device,
                **kwargs
            )
        elif model_type == "openai-compatible":
            from .openai import OpenAICompatibleEmbedding
            return OpenAICompatibleEmbedding(
                api_url=kwargs.get("api_url", "http://localhost:8000/v1"),
                api_key=kwargs.get("api_key"),
                model_name=model_name or "text-embedding-ada-002",
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}") 