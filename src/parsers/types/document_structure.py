from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .document_element import DocumentElement
from .element_type import ElementType


@dataclass
class DocumentStructure:
    """Represents the hierarchical structure of a document"""
    elements: List[DocumentElement]
    reading_order: List[str]  # Element IDs in reading order
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    page_dimensions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    
    def get_elements_by_type(self, element_type: ElementType) -> List[DocumentElement]:
        """Get all elements of a specific type"""
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_elements_by_page(self, page: int) -> List[DocumentElement]:
        """Get all elements on a specific page"""
        return [e for e in self.elements if e.bbox.page == page]
    
    def get_reading_order_text(self) -> str:
        """Get text content in reading order"""
        element_dict = {e.element_id: e for e in self.elements}
        ordered_text = []
        for element_id in self.reading_order:
            if element_id in element_dict:
                element = element_dict[element_id]
                if element.element_type in [ElementType.TEXT, ElementType.PARAGRAPH, ElementType.HEADING]:
                    ordered_text.append(element.content)
        return "\n".join(ordered_text)