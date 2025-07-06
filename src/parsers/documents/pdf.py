from typing import Dict, List
import pdfplumber

from .base import DocumentParser
from ..layout_analyser import LayoutAnalyzer
from ..types import BoundingBox, DocumentElement, DocumentStructure, ElementType, ParsedDocument

# Advanced layout analysis
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

class PositionalPDFParser(DocumentParser):
    """Enhanced PDF parser that preserves spatial information"""
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')
    
    def parse(self, file_path: str) -> ParsedDocument:
        elements = []
        metadata = {}
        errors = []
        page_dimensions = {}
        
        try:
            if PYMUPDF_AVAILABLE:
                # Use PyMuPDF for better layout analysis
                doc = fitz.open(file_path)
                
                metadata = {
                    'pages': len(doc),
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'creator': doc.metadata.get('creator', ''),
                    'creation_date': doc.metadata.get('creationDate', ''),
                }
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_dimensions[page_num] = (page.rect.width, page.rect.height)
                    
                    # Get text blocks with positions
                    blocks = page.get_text("dict")
                    
                    for block in blocks["blocks"]:
                        if "lines" in block:  # Text block
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    bbox = BoundingBox(
                                        x0=span["bbox"][0],
                                        y0=span["bbox"][1],
                                        x1=span["bbox"][2],
                                        y1=span["bbox"][3],
                                        page=page_num
                                    )
                                    
                                    # Determine element type based on font size and style
                                    element_type = self._classify_text_element(span)
                                    
                                    element = DocumentElement(
                                        content=span["text"],
                                        element_type=element_type,
                                        bbox=bbox,
                                        style={
                                            'font': span["font"],
                                            'size': span["size"],
                                            'flags': span["flags"],
                                            'color': span["color"]
                                        }
                                    )
                                    elements.append(element)
                    
                    # Extract tables with positions
                    tables = page.find_tables()
                    for table in tables:
                        bbox = BoundingBox(
                            x0=table.bbox[0],
                            y0=table.bbox[1],
                            x1=table.bbox[2],
                            y1=table.bbox[3],
                            page=page_num
                        )
                        
                        table_data = table.extract()
                        table_content = "\n".join(["\t".join(row) for row in table_data])
                        
                        element = DocumentElement(
                            content=table_content,
                            element_type=ElementType.TABLE,
                            bbox=bbox,
                            metadata={'table_data': table_data}
                        )
                        elements.append(element)
                    
                    # Extract images with positions
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        img_rect = page.get_image_bbox(img[7])  # Get image bbox
                        bbox = BoundingBox(
                            x0=img_rect.x0,
                            y0=img_rect.y0,
                            x1=img_rect.x1,
                            y1=img_rect.y1,
                            page=page_num
                        )
                        
                        element = DocumentElement(
                            content=f"[Image {img_index + 1}]",
                            element_type=ElementType.IMAGE,
                            bbox=bbox,
                            metadata={'image_index': img_index}
                        )
                        elements.append(element)
                
                doc.close()
                
            else:
                # Fallback to pdfplumber with basic positioning
                with pdfplumber.open(file_path) as pdf:
                    metadata = {
                        'pages': len(pdf.pages),
                        'title': pdf.metadata.get('Title', ''),
                        'author': pdf.metadata.get('Author', ''),
                    }
                    
                    for page_num, page in enumerate(pdf.pages):
                        page_dimensions[page_num] = (page.width, page.height)
                        
                        # Extract text with positions
                        chars = page.chars
                        if chars:
                            # Group characters into words and lines
                            words = self._group_chars_into_words(chars)
                            for word_chars in words:
                                if word_chars:
                                    bbox = BoundingBox(
                                        x0=min(c['x0'] for c in word_chars),
                                        y0=min(c['top'] for c in word_chars),
                                        x1=max(c['x1'] for c in word_chars),
                                        y1=max(c['bottom'] for c in word_chars),
                                        page=page_num
                                    )
                                    
                                    text = ''.join(c['text'] for c in word_chars)
                                    element = DocumentElement(
                                        content=text,
                                        element_type=ElementType.TEXT,
                                        bbox=bbox,
                                        style={
                                            'font': word_chars[0].get('fontname', ''),
                                            'size': word_chars[0].get('size', 0)
                                        }
                                    )
                                    elements.append(element)
                        
                        # Extract tables
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables):
                            # Estimate table position (pdfplumber doesn't provide exact coords)
                            bbox = BoundingBox(
                                x0=0, y0=0, x1=page.width, y1=page.height,
                                page=page_num
                            )
                            
                            table_content = "\n".join(["\t".join(str(cell) for cell in row) for row in table])
                            element = DocumentElement(
                                content=table_content,
                                element_type=ElementType.TABLE,
                                bbox=bbox,
                                metadata={'table_data': table, 'table_index': table_idx}
                            )
                            elements.append(element)
        
        except Exception as e:
            errors.append(f"PDF parsing error: {str(e)}")
        
        # Determine reading order
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
            file_type="pdf",
            parser_used="PositionalPDFParser",
            errors=errors if errors else None
        )
    
    def _classify_text_element(self, span: Dict) -> ElementType:
        """Classify text element based on font properties"""
        size = span.get("size", 12)
        flags = span.get("flags", 0)
        
        # Check if bold (flag 16) or large font
        if flags & 16 or size > 16:
            return ElementType.HEADING
        elif size < 8:
            return ElementType.CAPTION
        else:
            return ElementType.TEXT
    
    def _group_chars_into_words(self, chars: List[Dict], threshold: float = 2.0) -> List[List[Dict]]:
        """Group characters into words based on spacing"""
        if not chars:
            return []
        
        words = []
        current_word = [chars[0]]
        
        for char in chars[1:]:
            # Check if character is close enough to be part of the same word
            if (abs(char['x0'] - current_word[-1]['x1']) <= threshold and
                abs(char['top'] - current_word[-1]['top']) <= threshold):
                current_word.append(char)
            else:
                if current_word:
                    words.append(current_word)
                current_word = [char]
        
        if current_word:
            words.append(current_word)
        
        return words