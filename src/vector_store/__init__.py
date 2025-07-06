from typing import Optional

from .base import VectorStore


class VectorStoreFactory:
    """Factory for creating vector stores"""
    
    @staticmethod
    def create_vector_store(
        store_type: str,
        collection_name: str = "document_collection",
        persist_directory: Optional[str] = None,
        host: str = "localhost",
        port: str = "19530",
        embedding_dimension: int = 768,
        recreate_collection: bool = False
    ) -> VectorStore:
        """
        Create a vector store of the specified type
        
        Args:
            store_type: Type of vector store ("chroma" or "milvus")
            collection_name: Name of the collection
            persist_directory: Directory to persist data (for Chroma)
            host: Milvus server host
            port: Milvus server port
            embedding_dimension: Dimension of embedding vectors
            recreate_collection: If True, drop and recreate the collection
            
        Returns:
            VectorStore instance
        """
        if store_type.lower() == "chroma":
            from .chroma import ChromaVectorStore
            return ChromaVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_dimension=embedding_dimension
            )
        elif store_type.lower() == "milvus":
            from .milvus import MilvusVectorStore
            return MilvusVectorStore(
                collection_name=collection_name,
                host=host,
                port=port,
                embedding_dimension=embedding_dimension,
                recreate_collection=recreate_collection
            )
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}") 