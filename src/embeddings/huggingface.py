import logging
from typing import List, Optional

from .base import EmbeddingModel

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class HuggingFaceEmbedding(EmbeddingModel):
    """Embedding model using HuggingFace Transformers"""
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        pooling_strategy: str = "mean",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16
    ):
        """
        Initialize the HuggingFace embedding model
        
        Args:
            model_name: Name of the HuggingFace model to use
            pooling_strategy: How to pool token embeddings ("mean", "cls", or "last")
            device: Device to run the model on (None for auto-detection)
            max_length: Maximum sequence length
            batch_size: Batch size for processing multiple documents
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is not installed. Install with 'pip install transformers torch'")
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Determine device
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Dimension is based on the model's hidden size
        self._embedding_dimension = self.model.config.hidden_size
        
        logger.info(f"Initialized HuggingFaceEmbedding with model '{model_name}' on device '{self.device}'")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of documents"""
        all_embeddings = []
        
        # Process in batches to avoid OOM
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a single batch of texts"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get the embeddings based on pooling strategy
            if self.pooling_strategy == "cls":
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0]
            elif self.pooling_strategy == "last":
                # Use last token embedding (excluding padding tokens)
                attention_mask = inputs["attention_mask"]
                last_token_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(outputs.last_hidden_state.size(0), device=self.device)
                embeddings = outputs.last_hidden_state[batch_indices, last_token_indices]
            else:
                # Use mean pooling (default)
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to list
            return embeddings.cpu().numpy().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query"""
        return self._embed_batch([text])[0]
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self._embedding_dimension

