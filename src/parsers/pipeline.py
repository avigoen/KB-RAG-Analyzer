import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from .documents.base import DocumentParser
from .documents.html import PositionalHTMLParser
from .documents.image import PositionalImageParser
from .layout_analyser import LayoutAnalyzer
from .documents.pdf import PositionalPDFParser
from .types import BoundingBox, DocumentElement, DocumentStructure, ElementType, ParsedDocument
from .documents.word import PositionalWordParser
from .documents.excel import PositionalExcelParser
from .documents.text import PositionalTextParser
from .documents.json import PositionalJSONParser
from .documents.xml import PositionalXMLParser
from .documents.csv import PositionalCSVParser


class PositionalDocumentParsingPipeline:
    """Enhanced pipeline with spatial awareness and layout analysis"""
    
    def __init__(self):
        self.parsers = [
            PositionalPDFParser(),
            PositionalWordParser(),
            PositionalHTMLParser(),
            PositionalImageParser(),
            # Keep original parsers for unsupported formats
            PositionalExcelParser(),
            PositionalTextParser(),
            PositionalJSONParser(),
            PositionalXMLParser(),
            PositionalCSVParser()
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_parser(self, file_path: str) -> Optional[DocumentParser]:
        """Get the appropriate parser for a file"""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None
    
    def parse_document(self, file_path: str) -> ParsedDocument:
        """Parse a single document with spatial awareness"""
        if not os.path.exists(file_path):
            return ParsedDocument(
                structure=DocumentStructure(elements=[], reading_order=[]),
                metadata={},
                file_path=file_path,
                file_type="unknown",
                parser_used="None",
                errors=[f"File not found: {file_path}"]
            )
        
        parser = self.get_parser(file_path)
        if not parser:
            mime_type, _ = mimetypes.guess_type(file_path)
            return ParsedDocument(
                structure=DocumentStructure(elements=[], reading_order=[]),
                metadata={'mime_type': mime_type},
                file_path=file_path,
                file_type="unsupported",
                parser_used="None",
                errors=[f"No parser available for file type: {Path(file_path).suffix}"]
            )
        
        self.logger.info(f"Parsing {file_path} with {parser.__class__.__name__}")
        result = parser.parse(file_path)
        
        # Enhance with layout analysis
        if result.structure.elements:
            result.structure.relationships = self._build_relationships(result.structure.elements)
        
        return result
    
    def _build_relationships(self, elements: List[DocumentElement]) -> Dict[str, List[str]]:
        """Build parent-child relationships between elements"""
        relationships = {}
        
        # Sort elements by position (top to bottom, left to right)
        sorted_elements = sorted(elements, key=lambda e: (e.bbox.page, e.bbox.y0, e.bbox.x0))
        
        for i, element in enumerate(sorted_elements):
            if element.element_type == ElementType.HEADING:
                # Find elements that belong to this heading
                children = []
                for j in range(i + 1, len(sorted_elements)):
                    next_element = sorted_elements[j]
                    # Stop at next heading of same or higher level
                    if next_element.element_type == ElementType.HEADING:
                        break
                    children.append(next_element.element_id)
                
                if children:
                    relationships[element.element_id] = children
        
        return relationships
    
    def analyze_layout(self, result: ParsedDocument) -> Dict[str, Any]:
        """Analyze document layout and provide insights"""
        analysis = {
            'total_elements': len(result.structure.elements),
            'element_types': {},
            'pages': len(result.structure.page_dimensions),
            'reading_flow': [],
            'column_detection': {},
            'spatial_relationships': {}
        }
        
        # Count element types
        for element in result.structure.elements:
            element_type = element.element_type.value
            analysis['element_types'][element_type] = analysis['element_types'].get(element_type, 0) + 1
        
        # Analyze reading flow per page
        for page_num in result.structure.page_dimensions:
            page_elements = result.structure.get_elements_by_page(page_num)
            if page_elements:
                # Detect columns
                columns = LayoutAnalyzer.detect_columns(page_elements)
                analysis['column_detection'][page_num] = {
                    'column_count': len(columns),
                    'elements_per_column': [len(col) for col in columns]
                }
                
                # Analyze reading flow
                flow_analysis = self._analyze_reading_flow(page_elements)
                analysis['reading_flow'].append({
                    'page': page_num,
                    'flow_type': flow_analysis['type'],
                    'confidence': flow_analysis['confidence']
                })
        
        # Analyze spatial relationships
        analysis['spatial_relationships'] = self._analyze_spatial_relationships(result.structure.elements)
        
        return analysis
    
    def _analyze_reading_flow(self, elements: List[DocumentElement]) -> Dict[str, Any]:
        """Analyze the reading flow pattern of elements"""
        if len(elements) < 3:
            return {'type': 'unknown', 'confidence': 0.0}
        
        # Sort elements by reading order
        sorted_elements = sorted(elements, key=lambda e: (e.bbox.y0, e.bbox.x0))
        
        # Analyze if it's single column, multi-column, or complex layout
        x_positions = [e.bbox.center_x for e in sorted_elements]
        y_positions = [e.bbox.center_y for e in sorted_elements]
        
        # Check for column patterns
        x_clusters = self._find_clusters(x_positions)
        
        if len(x_clusters) == 1:
            return {'type': 'single_column', 'confidence': 0.9}
        elif len(x_clusters) == 2:
            return {'type': 'two_column', 'confidence': 0.8}
        elif len(x_clusters) > 2:
            return {'type': 'multi_column', 'confidence': 0.7}
        else:
            return {'type': 'complex', 'confidence': 0.5}
    
    def _find_clusters(self, positions: List[float], threshold: float = 50.0) -> List[List[float]]:
        """Find clusters in position data"""
        if not positions:
            return []
        
        sorted_positions = sorted(positions)
        clusters = [[sorted_positions[0]]]
        
        for pos in sorted_positions[1:]:
            if pos - clusters[-1][-1] <= threshold:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        return clusters
    
    def _analyze_spatial_relationships(self, elements: List[DocumentElement]) -> Dict[str, Any]:
        """Analyze spatial relationships between elements"""
        relationships = {
            'overlapping_elements': [],
            'nested_elements': [],
            'adjacent_elements': [],
            'alignment_groups': []
        }
        
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                if elem1.bbox.page != elem2.bbox.page:
                    continue
                
                # Check for overlapping elements
                if elem1.bbox.overlaps(elem2.bbox):
                    relationships['overlapping_elements'].append({
                        'element1': elem1.element_id,
                        'element2': elem2.element_id,
                        'overlap_type': 'contains' if elem1.bbox.contains(elem2.bbox) else 'intersects'
                    })
                
                # Check for nested elements
                if elem1.bbox.contains(elem2.bbox):
                    relationships['nested_elements'].append({
                        'parent': elem1.element_id,
                        'child': elem2.element_id
                    })
                
                # Check for adjacent elements
                if self._are_adjacent(elem1.bbox, elem2.bbox):
                    relationships['adjacent_elements'].append({
                        'element1': elem1.element_id,
                        'element2': elem2.element_id,
                        'adjacency_type': self._get_adjacency_type(elem1.bbox, elem2.bbox)
                    })
        
        # Find alignment groups
        relationships['alignment_groups'] = self._find_alignment_groups(elements)
        
        return relationships
    
    def _are_adjacent(self, bbox1: BoundingBox, bbox2: BoundingBox, threshold: float = 10.0) -> bool:
        """Check if two bounding boxes are adjacent"""
        # Check horizontal adjacency
        if (abs(bbox1.x1 - bbox2.x0) <= threshold or abs(bbox2.x1 - bbox1.x0) <= threshold):
            # Check vertical overlap
            if not (bbox1.y1 < bbox2.y0 or bbox2.y1 < bbox1.y0):
                return True
        
        # Check vertical adjacency
        if (abs(bbox1.y1 - bbox2.y0) <= threshold or abs(bbox2.y1 - bbox1.y0) <= threshold):
            # Check horizontal overlap
            if not (bbox1.x1 < bbox2.x0 or bbox2.x1 < bbox1.x0):
                return True
        
        return False
    
    def _get_adjacency_type(self, bbox1: BoundingBox, bbox2: BoundingBox) -> str:
        """Determine the type of adjacency between two bounding boxes"""
        if abs(bbox1.x1 - bbox2.x0) <= abs(bbox1.y1 - bbox2.y0):
            return 'horizontal'
        else:
            return 'vertical'
    
    def _find_alignment_groups(self, elements: List[DocumentElement]) -> List[Dict[str, Any]]:
        """Find groups of elements that are aligned"""
        alignment_groups = []
        
        # Group by horizontal alignment (same Y coordinate)
        y_groups = {}
        for element in elements:
            y_key = round(element.bbox.center_y / 5) * 5  # Group within 5 pixels
            if y_key not in y_groups:
                y_groups[y_key] = []
            y_groups[y_key].append(element)
        
        for y_coord, group in y_groups.items():
            if len(group) > 1:
                alignment_groups.append({
                    'type': 'horizontal',
                    'elements': [e.element_id for e in group],
                    'coordinate': y_coord
                })
        
        # Group by vertical alignment (same X coordinate)
        x_groups = {}
        for element in elements:
            x_key = round(element.bbox.center_x / 5) * 5  # Group within 5 pixels
            if x_key not in x_groups:
                x_groups[x_key] = []
            x_groups[x_key].append(element)
        
        for x_coord, group in x_groups.items():
            if len(group) > 1:
                alignment_groups.append({
                    'type': 'vertical',
                    'elements': [e.element_id for e in group],
                    'coordinate': x_coord
                })
        
        return alignment_groups
    
    def extract_structured_content(self, result: ParsedDocument) -> Dict[str, Any]:
        """Extract structured content maintaining spatial context"""
        structured = {
            'title': '',
            'sections': [],
            'tables': [],
            'images': [],
            'metadata': result.metadata
        }
        
        # Extract title (usually first heading)
        headings = result.structure.get_elements_by_type(ElementType.HEADING)
        if headings:
            structured['title'] = headings[0].content
        
        # Extract sections with spatial context
        current_section = None
        
        for element_id in result.structure.reading_order:
            element = next((e for e in result.structure.elements if e.element_id == element_id), None)
            if not element:
                continue
            
            if element.element_type == ElementType.HEADING:
                if current_section:
                    structured['sections'].append(current_section)
                
                current_section = {
                    'title': element.content,
                    'content': [],
                    'position': {
                        'page': element.bbox.page,
                        'bbox': {
                            'x0': element.bbox.x0,
                            'y0': element.bbox.y0,
                            'x1': element.bbox.x1,
                            'y1': element.bbox.y1
                        }
                    }
                }
            
            elif element.element_type in [ElementType.TEXT, ElementType.PARAGRAPH]:
                if current_section:
                    current_section['content'].append({
                        'text': element.content,
                        'position': {
                            'page': element.bbox.page,
                            'bbox': {
                                'x0': element.bbox.x0,
                                'y0': element.bbox.y0,
                                'x1': element.bbox.x1,
                                'y1': element.bbox.y1
                            }
                        }
                    })
            
            elif element.element_type == ElementType.TABLE:
                structured['tables'].append({
                    'content': element.content,
                    'position': {
                        'page': element.bbox.page,
                        'bbox': {
                            'x0': element.bbox.x0,
                            'y0': element.bbox.y0,
                            'x1': element.bbox.x1,
                            'y1': element.bbox.y1
                        }
                    },
                    'metadata': element.metadata
                })
            
            elif element.element_type == ElementType.IMAGE:
                structured['images'].append({
                    'content': element.content,
                    'position': {
                        'page': element.bbox.page,
                        'bbox': {
                            'x0': element.bbox.x0,
                            'y0': element.bbox.y0,
                            'x1': element.bbox.x1,
                            'y1': element.bbox.y1
                        }
                    },
                    'metadata': element.metadata
                })
        
        if current_section:
            structured['sections'].append(current_section)
        
        return structured
    
    def export_with_positions(self, results: List[ParsedDocument], output_path: str, format: str = 'json'):
        """Export results with spatial information preserved"""
        if format.lower() == 'json':
            export_data = []
            for result in results:
                # Include layout analysis
                layout_analysis = self.analyze_layout(result)
                structured_content = self.extract_structured_content(result)
                
                export_data.append({
                    'file_path': result.file_path,
                    'file_type': result.file_type,
                    'parser_used': result.parser_used,
                    'metadata': result.metadata,
                    'layout_analysis': layout_analysis,
                    'structured_content': structured_content,
                    'elements': [
                        {
                            'id': elem.element_id,
                            'type': elem.element_type.value,
                            'content': elem.content,
                            'bbox': {
                                'x0': elem.bbox.x0,
                                'y0': elem.bbox.y0,
                                'x1': elem.bbox.x1,
                                'y1': elem.bbox.y1,
                                'page': elem.bbox.page
                            },
                            'style': elem.style,
                            'confidence': elem.confidence,
                            'metadata': elem.metadata
                        } for elem in result.structure.elements
                    ],
                    'reading_order': result.structure.reading_order,
                    'page_dimensions': result.structure.page_dimensions,
                    'errors': result.errors
                })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'html':
            # Export as HTML with visual representation
            html_content = self._generate_html_visualization(results)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
    
    def _generate_html_visualization(self, results: List[ParsedDocument]) -> str:
        """Generate HTML visualization of document layout"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Document Layout Analysis</title>",
            "<style>",
            ".document { border: 1px solid #ccc; margin: 20px; padding: 20px; }",
            ".page { position: relative; border: 1px solid #ddd; margin: 10px 0; }",
            ".element { position: absolute; border: 1px solid #666; background: rgba(0,0,0,0.1); }",
            ".heading { background: rgba(255,0,0,0.2); }",
            ".text { background: rgba(0,255,0,0.2); }",
            ".table { background: rgba(0,0,255,0.2); }",
            ".image { background: rgba(255,255,0,0.2); }",
            "</style>",
            "</head><body>"
        ]
        
        for result in results:
            html_parts.append(f"<div class='document'>")
            html_parts.append(f"<h2>{result.file_path}</h2>")
            
            # Group elements by page
            for page_num in result.structure.page_dimensions:
                page_elements = result.structure.get_elements_by_page(page_num)
                if page_elements:
                    width, height = result.structure.page_dimensions[page_num]
                    html_parts.append(f"<div class='page' style='width: {width*0.5}px; height: {height*0.5}px;'>")
                    html_parts.append(f"<h3>Page {page_num + 1}</h3>")
                    
                    for element in page_elements:
                        css_class = element.element_type.value
                        style = (f"left: {element.bbox.x0*0.5}px; top: {element.bbox.y0*0.5}px; "
                                f"width: {element.bbox.width*0.5}px; height: {element.bbox.height*0.5}px;")
                        title = element.content[:50] + "..." if len(element.content) > 50 else element.content
                        html_parts.append(f"<div class='element {css_class}' style='{style}' title='{title}'></div>")
                    
                    html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        html_parts.append("</body></html>")
        return "\n".join(html_parts)
    
    def get_contextual_content(self, result: ParsedDocument, query_bbox: BoundingBox, 
                              context_radius: float = 100.0) -> List[DocumentElement]:
        """Get content within a spatial context of a query region"""
        contextual_elements = []
        
        for element in result.structure.elements:
            if element.bbox.page != query_bbox.page:
                continue
            
            # Calculate distance from query region
            distance = self._calculate_distance(element.bbox, query_bbox)
            
            if distance <= context_radius:
                contextual_elements.append(element)
        
        # Sort by distance
        contextual_elements.sort(key=lambda e: self._calculate_distance(e.bbox, query_bbox))
        
        return contextual_elements
    
    def _calculate_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate distance between two bounding boxes"""
        center1_x, center1_y = bbox1.center_x, bbox1.center_y
        center2_x, center2_y = bbox2.center_x, bbox2.center_y
        
        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
