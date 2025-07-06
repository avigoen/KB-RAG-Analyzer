from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents spatial coordinates of an element"""
    x0: float  # left
    y0: float  # top
    x1: float  # right
    y1: float  # bottom
    page: int = 0
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2
    
    def overlaps(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box overlaps with another"""
        return not (self.x1 < other.x0 or self.x0 > other.x1 or 
                   self.y1 < other.y0 or self.y0 > other.y1)
    
    def contains(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box contains another"""
        return (self.x0 <= other.x0 and self.y0 <= other.y0 and
                self.x1 >= other.x1 and self.y1 >= other.y1)