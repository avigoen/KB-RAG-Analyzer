from dataclasses import dataclass
from typing import Any, Dict, List

from .bounding_box import BoundingBox


@dataclass
class CSVCell:
    """Represents a single cell in the CSV with position information"""
    value: str
    row: int
    column: int
    column_name: str
    bbox: BoundingBox
    data_type: str = "string"
    is_header: bool = False
    is_numeric: bool = False
    is_empty: bool = False
    
    def __post_init__(self):
        # Determine data type and properties
        self.is_empty = not self.value or str(self.value).strip() == ""
        if not self.is_empty:
            self.data_type = self._infer_data_type(self.value)
            self.is_numeric = self.data_type in ["integer", "float"]
    
    def _infer_data_type(self, value: str) -> str:
        """Infer the data type of a cell value"""
        if not value or str(value).strip() == "":
            return "empty"
        
        value_str = str(value).strip()
        
        # Try integer
        try:
            int(value_str)
            return "integer"
        except ValueError:
            pass
        
        # Try float
        try:
            float(value_str)
            return "float"
        except ValueError:
            pass
        
        # Check for date patterns
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value_str):
                return "date"
        
        # Check for boolean
        if value_str.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
            return "boolean"
        
        # Default to string
        return "string"

@dataclass
class CSVColumn:
    """Represents a column in the CSV with metadata"""
    name: str
    index: int
    data_types: List[str]
    unique_values: int
    null_count: int
    max_length: int
    bbox: BoundingBox
    statistical_summary: Dict[str, Any]

@dataclass
class CSVRow:
    """Represents a row in the CSV with metadata"""
    index: int
    cells: List[CSVCell]
    bbox: BoundingBox
    is_header: bool = False
    completeness: float = 0.0  # Percentage of non-empty cells
    
    def __post_init__(self):
        # Calculate completeness
        non_empty_cells = sum(1 for cell in self.cells if not cell.is_empty)
        self.completeness = non_empty_cells / len(self.cells) if self.cells else 0.0
