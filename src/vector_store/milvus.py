import logging
from typing import Any, Dict, List, Optional
import uuid

from .base import VectorStore

try:
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
    MILVUS_AVAILABLE = False
except ImportError:
    MILVUS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MilvusVectorStore(VectorStore):
    """Milvus vector database implementation"""
    
    def __init__(
        self,
        collection_name: str = "document_collection",
        host: str = "localhost",
        port: str = "19530",
        embedding_dimension: int = 768,
        recreate_collection: bool = False
    ):
        """
        Initialize Milvus vector store
        
        Args:
            collection_name: Name of the collection
            host: Milvus server host
            port: Milvus server port
            embedding_dimension: Dimension of embedding vectors
            recreate_collection: If True, drop and recreate the collection
        """
        if not MILVUS_AVAILABLE:
            raise ImportError("PyMilvus is not installed. Install with 'pip install pymilvus'")
        
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Connect to Milvus
        connections.connect(host=host, port=port)
        
        # Create collection if it doesn't exist or if recreate_collection is True
        if Collection.has_collection(collection_name) and recreate_collection:
            Collection(collection_name).drop()
        
        if not Collection.has_collection(collection_name):
            self._create_collection()
        
        self.collection = Collection(collection_name)
        self.collection.load()
        
        logger.info(f"Initialized MilvusVectorStore with collection '{collection_name}'")
    
    def _create_collection(self):
        """Create a new Milvus collection with the appropriate schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields=fields, description=f"Document collection for RAG")
        collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index on the embedding field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """Add documents with embeddings to Milvus"""
        if len(documents) != len(embeddings):
            raise ValueError(f"Number of documents ({len(documents)}) and embeddings ({len(embeddings)}) must match")
        
        # Prepare document data for Milvus
        ids = [doc.get('chunk_id', str(uuid.uuid4())) for doc in documents]
        texts = [doc['text'] for doc in documents]
        
        # Convert document metadata to JSON compatible format
        metadatas = []
        for doc in documents:
            metadata = {k: v for k, v in doc.items() if k != 'text' and k != 'chunk_id'}
            metadatas.append(metadata)
        
        # Insert data
        data = [ids, embeddings, texts, metadatas]
        self.collection.insert(data)
        
        return ids
    
    def get_documents_by_id(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Get documents by their IDs from Milvus"""
        # Query by ID
        self.collection.flush()
        results = self.collection.query(
            expr=f'id in ["{", ".join(ids)}"]',
            output_fields=["id", "text", "metadata"]
        )
        
        documents = []
        for result in results:
            doc = {
                'chunk_id': result['id'],
                'text': result['text']
            }
            # Add metadata
            if result.get('metadata'):
                doc.update(result['metadata'])
            documents.append(doc)
        
        return documents
    
    def query_by_embedding(self, 
                          embedding: List[float], 
                          k: int = 5, 
                          filter_criteria: Optional[Dict[str, Any]] = None
                          ) -> List[Dict[str, Any]]:
        """Query Milvus using an embedding vector"""
        # Convert filter criteria to Milvus expression format
        expr = None
        if filter_criteria:
            conditions = []
            for key, value in filter_criteria.items():
                if isinstance(value, str):
                    conditions.append(f'metadata["{key}"] == "{value}"')
                else:
                    conditions.append(f'metadata["{key}"] == {value}')
            if conditions:
                expr = " && ".join(conditions)
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 32}
        }
        
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["id", "text", "metadata"]
        )
        
        documents = []
        for hit in results[0]:
            doc = {
                'chunk_id': hit.id,
                'text': hit.entity.get('text'),
                'score': float(hit.score)  # Already a similarity score
            }
            # Add metadata
            if hit.entity.get('metadata'):
                doc.update(hit.entity.get('metadata'))
            documents.append(doc)
        
        return documents
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by their IDs from Milvus"""
        expr = f'id in ["{", ".join(ids)}"]'
        self.collection.delete(expr)
    
    def count(self) -> int:
        """Count documents in the Milvus collection"""
        return self.collection.num_entities