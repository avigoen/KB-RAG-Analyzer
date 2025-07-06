from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .document_structure import DocumentStructure
from .element_type import ElementType


@dataclass
class ParsedDocument:
    """Enhanced container for parsed document content with spatial information"""
    structure: DocumentStructure
    metadata: Dict[str, Any]
    file_path: str
    file_type: str
    parser_used: str
    raw_content: str = ""  # Fallback linear text
    errors: Optional[List[str]] = None
    
    @property
    def content(self) -> str:
        """Get content respecting document structure"""
        return self.structure.get_reading_order_text() or self.raw_content
    
    @property
    def tables(self) -> List[Dict]:
        """Get all tables with their positions"""
        tables = []
        for element in self.structure.get_elements_by_type(ElementType.TABLE):
            tables.append({
                'content': element.content,
                'bbox': element.bbox,
                'page': element.bbox.page,
                'metadata': element.metadata
            })
        return tables
    
    @property
    def images(self) -> List[Dict]:
        """Get all images with their positions"""
        images = []
        for element in self.structure.get_elements_by_type(ElementType.IMAGE):
            images.append({
                'content': element.content,
                'bbox': element.bbox,
                'page': element.bbox.page,
                'metadata': element.metadata
            })
        return images