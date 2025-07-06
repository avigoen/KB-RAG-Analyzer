import json
from typing import Any, Dict, List, Tuple

from .base import DocumentParser
from ..layout_analyser import LayoutAnalyzer
from ..types import BoundingBox, DocumentElement, DocumentStructure, ElementType, ParsedDocument


class PositionalJSONParser(DocumentParser):
    """Enhanced JSON parser that preserves hierarchical structure and positioning"""
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith('.json')
    
    def parse(self, file_path: str) -> ParsedDocument:
        elements = []
        metadata = {}
        errors = []
        page_dimensions = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse JSON
            data = json.loads(content)
            
            # Analyze JSON structure
            structure_analysis = self._analyze_json_structure(data)
            metadata.update(structure_analysis)
            
            metadata.update({
                'file_size': len(content),
                'is_array': isinstance(data, list),
                'is_object': isinstance(data, dict)
            })
            
            # Convert JSON to positioned elements
            y_position = 0
            max_width = 0
            
            elements, y_position, max_width = self._process_json_value(
                data, "", 0, 0, 0, elements
            )
            
            page_dimensions[0] = (max_width, y_position)
            
        except json.JSONDecodeError as e:
            errors.append(f"JSON parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"JSON processing error: {str(e)}")
        
        reading_order = LayoutAnalyzer.determine_reading_order(elements)
        
        structure = DocumentStructure(
            elements=elements,
            reading_order=reading_order,
            page_dimensions=page_dimensions
        )
        
        return ParsedDocument(
            structure=structure,
            metadata=metadata,
            file_path=file_path,
            file_type="json",
            parser_used="PositionalJSONParser",
            errors=errors if errors else None
        )
    
    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure recursively"""
        analysis = {
            'max_depth': 0,
            'total_keys': 0,
            'total_values': 0,
            'data_types': {},
            'array_sizes': []
        }
        
        def analyze_recursive(obj, depth=0):
            analysis['max_depth'] = max(analysis['max_depth'], depth)
            
            if isinstance(obj, dict):
                analysis['total_keys'] += len(obj)
                for key, value in obj.items():
                    analysis['total_values'] += 1
                    type_name = type(value).__name__
                    analysis['data_types'][type_name] = analysis['data_types'].get(type_name, 0) + 1
                    analyze_recursive(value, depth + 1)
            elif isinstance(obj, list):
                analysis['array_sizes'].append(len(obj))
                for item in obj:
                    analyze_recursive(item, depth + 1)
            else:
                type_name = type(obj).__name__
                analysis['data_types'][type_name] = analysis['data_types'].get(type_name, 0) + 1
        
        analyze_recursive(data)
        return analysis
    
    def _process_json_value(self, value: Any, key: str, x_offset: int, y_offset: int, 
                           depth: int, elements: List[DocumentElement]) -> Tuple[List[DocumentElement], int, int]:
        """Process JSON value recursively and create positioned elements"""
        max_width = x_offset
        
        if isinstance(value, dict):
            # Create element for object
            if key:
                bbox = BoundingBox(
                    x0=x_offset,
                    y0=y_offset,
                    x1=x_offset + len(key) * 8 + 20,
                    y1=y_offset + 20,
                    page=0
                )
                
                element = DocumentElement(
                    content=f"{key}: {{",
                    element_type=ElementType.HEADING,
                    bbox=bbox,
                    metadata={
                        'json_type': 'object_key',
                        'key': key,
                        'depth': depth,
                        'object_size': len(value)
                    }
                )
                elements.append(element)
                max_width = max(max_width, bbox.x1)
                y_offset += 25
            
            # Process each key-value pair
            for obj_key, obj_value in value.items():
                elements, y_offset, width = self._process_json_value(
                    obj_value, obj_key, x_offset + 20, y_offset, depth + 1, elements
                )
                max_width = max(max_width, width)
        
        elif isinstance(value, list):
            # Create element for array
            if key:
                bbox = BoundingBox(
                    x0=x_offset,
                    y0=y_offset,
                    x1=x_offset + len(key) * 8 + 20,
                    y1=y_offset + 20,
                    page=0
                )
                
                element = DocumentElement(
                    content=f"{key}: [",
                    element_type=ElementType.HEADING,
                    bbox=bbox,
                    metadata={
                        'json_type': 'array_key',
                        'key': key,
                        'depth': depth,
                        'array_size': len(value)
                    }
                )
                elements.append(element)
                max_width = max(max_width, bbox.x1)
                y_offset += 25
            
            # Process each array item
            for i, item in enumerate(value):
                elements, y_offset, width = self._process_json_value(
                    item, f"[{i}]", x_offset + 20, y_offset, depth + 1, elements
                )
                max_width = max(max_width, width)
        
        else:
            # Create element for primitive value
            content = f"{key}: {json.dumps(value)}" if key else str(value)
            
            bbox = BoundingBox(
                x0=x_offset,
                y0=y_offset,
                x1=x_offset + len(content) * 8,
                y1=y_offset + 20,
                page=0
            )
            
            element = DocumentElement(
                content=content,
                element_type=ElementType.TEXT,
                bbox=bbox,
                metadata={
                    'json_type': 'primitive',
                    'key': key,
                    'value': value,
                    'value_type': type(value).__name__,
                    'depth': depth
                }
            )
            elements.append(element)
            max_width = max(max_width, bbox.x1)
            y_offset += 25
        
        return elements, y_offset, max_width