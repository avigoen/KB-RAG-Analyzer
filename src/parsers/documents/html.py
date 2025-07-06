from bs4 import BeautifulSoup

from .base import DocumentParser
from ..layout_analyser import LayoutAnalyzer
from ..types import BoundingBox, DocumentElement, DocumentStructure, ElementType, ParsedDocument


class PositionalHTMLParser(DocumentParser):
    """Enhanced HTML parser that preserves DOM structure and positioning"""
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.html', '.htm'))
    
    def parse(self, file_path: str) -> ParsedDocument:
        elements = []
        metadata = {}
        errors = []
        page_dimensions = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            metadata = {
                'title': soup.title.string if soup.title else '',
                'links': len(soup.find_all('a')),
                'images': len(soup.find_all('img')),
                'tables': len(soup.find_all('table'))
            }
            
            y_position = 0
            
            # Process elements maintaining hierarchy
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'table', 'ul', 'ol', 'img']):
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    element_type = ElementType.HEADING
                elif element.name == 'p':
                    element_type = ElementType.PARAGRAPH
                elif element.name in ['ul', 'ol']:
                    element_type = ElementType.LIST
                elif element.name == 'table':
                    element_type = ElementType.TABLE
                elif element.name == 'img':
                    element_type = ElementType.IMAGE
                else:
                    element_type = ElementType.TEXT
                
                text_content = element.get_text(strip=True)
                if text_content or element.name == 'img':
                    bbox = BoundingBox(
                        x0=0,
                        y0=y_position,
                        x1=800,  # Estimated page width
                        y1=y_position + 20,
                        page=0
                    )
                    
                    doc_element = DocumentElement(
                        content=text_content or f"[{element.name.upper()}]",
                        element_type=element_type,
                        bbox=bbox,
                        style={
                            'tag': element.name,
                            'class': element.get('class', []),
                            'id': element.get('id', '')
                        }
                    )
                    elements.append(doc_element)
                    y_position += 25
            
            page_dimensions[0] = (800, y_position)
            
        except Exception as e:
            errors.append(f"HTML parsing error: {str(e)}")
        
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
            file_type="html",
            parser_used="PositionalHTMLParser",
            errors=errors if errors else None
        )