"""
Document chunking strategies for optimal RAG pipeline performance.
"""
from typing import Dict, List, Optional, Tuple, Any
import re
from .types import ParsedDocument, DocumentElement, ElementType


class DocumentChunker:
    """
    Handles document chunking for RAG pipeline with various strategies:
    - Fixed-size chunks with overlap
    - Semantic chunking (paragraph/section-based)
    - Hierarchical chunking (preserves document structure)
    """
    
    def __init__(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50,
        strategy: str = "fixed",
        preserve_metadata: bool = True
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Target size of each chunk in tokens/chars
            chunk_overlap: Overlap between chunks in tokens/chars
            strategy: Chunking strategy - "fixed", "semantic", or "hierarchical"
            preserve_metadata: Whether to preserve metadata in chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.preserve_metadata = preserve_metadata
    
    def chunk_document(self, doc: ParsedDocument) -> List[Dict[str, Any]]:
        """
        Chunk a parsed document based on the selected strategy.
        
        Args:
            doc: The parsed document to chunk
            
        Returns:
            List of chunks, each containing text content and metadata
        """
        if self.strategy == "fixed":
            return self._fixed_size_chunking(doc)
        elif self.strategy == "semantic":
            return self._semantic_chunking(doc)
        elif self.strategy == "hierarchical":
            return self._hierarchical_chunking(doc)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _fixed_size_chunking(self, doc: ParsedDocument) -> List[Dict[str, Any]]:
        """
        Create fixed-size chunks with overlap.
        
        Args:
            doc: The parsed document
            
        Returns:
            List of chunks with text and metadata
        """
        text = doc.content
        chunks = []
        
        # Simple token count estimation (approximate)
        tokens = re.findall(r'\S+|\n', text)
        
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Get the chunk text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = ' '.join(chunk_tokens)
            
            # Create chunk with metadata
            chunk = {
                'text': chunk_text,
                'chunk_id': f"{doc.file_path}-chunk-{len(chunks)}",
                'start_idx': start_idx,
                'end_idx': end_idx,
                'document_id': doc.file_path
            }
            
            # Add metadata if needed
            if self.preserve_metadata:
                chunk['metadata'] = {
                    'source': doc.file_path,
                    'file_type': doc.file_type,
                    'page_range': self._estimate_page_range(doc, start_idx, end_idx),
                    'document_metadata': doc.metadata
                }
                
            chunks.append(chunk)
            
            # Move start index for next chunk (with overlap)
            start_idx = end_idx - self.chunk_overlap
        
        return chunks
    
    def _semantic_chunking(self, doc: ParsedDocument) -> List[Dict[str, Any]]:
        """
        Create chunks based on semantic boundaries (paragraphs, sections).
        
        Args:
            doc: The parsed document
            
        Returns:
            List of chunks with text and metadata
        """
        chunks = []
        current_chunk = []
        current_size = 0
        current_elements = []
        
        # Group elements into semantic chunks
        for element in doc.structure.elements:
            # Skip non-text elements
            if element.element_type not in [ElementType.TEXT, ElementType.PARAGRAPH, ElementType.HEADING]:
                continue
                
            element_tokens = len(re.findall(r'\S+', element.content))
            
            # If adding this element would exceed chunk size and we already have content,
            # create a new chunk
            if current_size + element_tokens > self.chunk_size and current_size > 0:
                chunk_text = '\n'.join(current_chunk)
                
                chunk = {
                    'text': chunk_text,
                    'chunk_id': f"{doc.file_path}-chunk-{len(chunks)}",
                    'document_id': doc.file_path,
                    'elements': [e.element_id for e in current_elements]
                }
                
                # Add metadata if needed
                if self.preserve_metadata:
                    chunk['metadata'] = {
                        'source': doc.file_path,
                        'file_type': doc.file_type,
                        'page_range': self._get_page_range_from_elements(current_elements),
                        'element_types': [e.element_type.value for e in current_elements],
                        'document_metadata': doc.metadata
                    }
                    
                chunks.append(chunk)
                
                # Reset for next chunk (with overlap if heading)
                if element.element_type == ElementType.HEADING:
                    current_chunk = []
                    current_size = 0
                    current_elements = []
                else:
                    # Keep the last element for overlap
                    overlap_idx = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_idx:]
                    current_size = sum(len(re.findall(r'\S+', c)) for c in current_chunk)
                    current_elements = current_elements[overlap_idx:]
            
            # Add the current element
            current_chunk.append(element.content)
            current_size += element_tokens
            current_elements.append(element)
        
        # Add the final chunk if there's content
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            
            chunk = {
                'text': chunk_text,
                'chunk_id': f"{doc.file_path}-chunk-{len(chunks)}",
                'document_id': doc.file_path,
                'elements': [e.element_id for e in current_elements]
            }
            
            if self.preserve_metadata:
                chunk['metadata'] = {
                    'source': doc.file_path,
                    'file_type': doc.file_type,
                    'page_range': self._get_page_range_from_elements(current_elements),
                    'element_types': [e.element_type.value for e in current_elements],
                    'document_metadata': doc.metadata
                }
                
            chunks.append(chunk)
        
        return chunks
    
    def _hierarchical_chunking(self, doc: ParsedDocument) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks preserving document structure.
        
        Args:
            doc: The parsed document
            
        Returns:
            List of chunks with hierarchical structure
        """
        chunks = []
        
        # Group elements by their structural relationships
        element_map = {e.element_id: e for e in doc.structure.elements}
        
        # Find all headings
        headings = [e for e in doc.structure.elements if e.element_type == ElementType.HEADING]
        
        # If no headings found, fall back to semantic chunking
        if not headings:
            return self._semantic_chunking(doc)
        
        # Process each heading and its content as a chunk
        for heading in headings:
            chunk_elements = [heading]
            
            # Find child elements from relationships or use spatial proximity
            child_ids = doc.structure.relationships.get(heading.element_id, [])
            
            if child_ids:
                # Get child elements from relationships
                for child_id in child_ids:
                    if child_id in element_map:
                        chunk_elements.append(element_map[child_id])
            else:
                # Use spatial heuristics - elements after this heading and before next heading
                heading_idx = doc.structure.elements.index(heading)
                next_heading_idx = len(doc.structure.elements)
                
                for i, e in enumerate(doc.structure.elements[heading_idx+1:], start=heading_idx+1):
                    if e.element_type == ElementType.HEADING:
                        next_heading_idx = i
                        break
                
                chunk_elements.extend(doc.structure.elements[heading_idx+1:next_heading_idx])
            
            # Check if the chunk is too large and needs splitting
            chunk_text = '\n'.join(e.content for e in chunk_elements)
            tokens = len(re.findall(r'\S+', chunk_text))
            
            if tokens <= self.chunk_size:
                # Small enough for a single chunk
                chunk = {
                    'text': chunk_text,
                    'chunk_id': f"{doc.file_path}-chunk-{len(chunks)}",
                    'document_id': doc.file_path,
                    'elements': [e.element_id for e in chunk_elements],
                    'heading': heading.content
                }
                
                if self.preserve_metadata:
                    chunk['metadata'] = {
                        'source': doc.file_path,
                        'file_type': doc.file_type,
                        'page_range': self._get_page_range_from_elements(chunk_elements),
                        'heading_id': heading.element_id,
                        'document_metadata': doc.metadata
                    }
                
                chunks.append(chunk)
            else:
                # Split large sections into sub-chunks
                subchunks = self._split_elements_into_chunks(chunk_elements, doc.file_path, len(chunks))
                
                for i, subchunk in enumerate(subchunks):
                    # Add heading to each subchunk for context
                    if i > 0:  # First chunk already has the heading
                        subchunk_text = f"{heading.content} (continued)\n{subchunk['text']}"
                    else:
                        subchunk_text = subchunk['text']
                        
                    subchunk['text'] = subchunk_text
                    subchunk['heading'] = heading.content
                    
                    chunks.append(subchunk)
        
        return chunks
    
    def _split_elements_into_chunks(
        self, elements: List[DocumentElement], file_path: str, chunk_offset: int
    ) -> List[Dict[str, Any]]:
        """
        Split a list of elements into chunks of appropriate size.
        
        Args:
            elements: List of document elements
            file_path: Path to the source document
            chunk_offset: Starting index for chunk IDs
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk = []
        current_elements = []
        current_size = 0
        
        for element in elements:
            element_tokens = len(re.findall(r'\S+', element.content))
            
            # If adding this element would exceed chunk size and we already have content,
            # create a new chunk
            if current_size + element_tokens > self.chunk_size and current_size > 0:
                chunk_text = '\n'.join(current_chunk)
                
                chunk = {
                    'text': chunk_text,
                    'chunk_id': f"{file_path}-chunk-{chunk_offset + len(chunks)}",
                    'document_id': file_path,
                    'elements': [e.element_id for e in current_elements]
                }
                
                # Add metadata if needed
                if self.preserve_metadata:
                    chunk['metadata'] = {
                        'source': file_path,
                        'page_range': self._get_page_range_from_elements(current_elements),
                        'element_types': [e.element_type.value for e in current_elements]
                    }
                
                chunks.append(chunk)
                
                # Reset with overlap
                overlap_idx = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_idx:]
                current_size = sum(len(re.findall(r'\S+', c)) for c in current_chunk)
                current_elements = current_elements[overlap_idx:]
            
            # Add the current element
            current_chunk.append(element.content)
            current_size += element_tokens
            current_elements.append(element)
        
        # Add the final chunk if there's content
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            
            chunk = {
                'text': chunk_text,
                'chunk_id': f"{file_path}-chunk-{chunk_offset + len(chunks)}",
                'document_id': file_path,
                'elements': [e.element_id for e in current_elements]
            }
            
            if self.preserve_metadata:
                chunk['metadata'] = {
                    'source': file_path,
                    'page_range': self._get_page_range_from_elements(current_elements),
                    'element_types': [e.element_type.value for e in current_elements]
                }
            
            chunks.append(chunk)
        
        return chunks
    
    def _estimate_page_range(self, doc: ParsedDocument, start_idx: int, end_idx: int) -> Tuple[int, int]:
        """
        Estimate the page range for a chunk based on token indices.
        This is an approximate method and works best for fixed chunking.
        
        Args:
            doc: The parsed document
            start_idx: Starting token index
            end_idx: Ending token index
            
        Returns:
            Tuple of (start_page, end_page)
        """
        # Default if we can't determine pages
        if not doc.metadata.get('pages'):
            return (0, 0)
            
        total_pages = doc.metadata.get('pages', 1)
        total_tokens = len(re.findall(r'\S+|\n', doc.content))
        
        if total_tokens == 0:
            return (0, 0)
            
        # Estimate pages based on token position (rough approximation)
        tokens_per_page = total_tokens / total_pages
        
        start_page = min(int(start_idx / tokens_per_page), total_pages - 1)
        end_page = min(int(end_idx / tokens_per_page), total_pages - 1)
        
        return (start_page, end_page)
    
    def _get_page_range_from_elements(self, elements: List[DocumentElement]) -> Tuple[int, int]:
        """
        Get the page range from a list of document elements.
        
        Args:
            elements: List of document elements
            
        Returns:
            Tuple of (start_page, end_page)
        """
        if not elements:
            return (0, 0)
            
        pages = [e.bbox.page for e in elements if hasattr(e, 'bbox') and e.bbox]
        
        if not pages:
            return (0, 0)
            
        return (min(pages), max(pages)) 