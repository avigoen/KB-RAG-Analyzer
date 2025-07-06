from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .bounding_box import BoundingBox
from .element_type import ElementType


@dataclass
class DocumentElement:
    """Represents a positioned element in a document"""
    content: str
    element_type: ElementType
    bbox: BoundingBox
    style: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    parent_id: Optional[str] = None
    element_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.element_id:
            self.element_id = f"{self.element_type.value}_{id(self)}"