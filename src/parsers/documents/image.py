from .base import DocumentParser
from ..layout_analyser import LayoutAnalyzer
from ..types import BoundingBox, DocumentElement, DocumentStructure, ElementType, ParsedDocument

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

class PositionalImageParser(DocumentParser):
    """Enhanced image parser with OCR and layout analysis"""
    
    def can_parse(self, file_path: str) -> bool:
        return (OCR_AVAILABLE and 
                file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')))
    
    def parse(self, file_path: str) -> ParsedDocument:
        elements = []
        metadata = {}
        errors = []
        page_dimensions = {}
        
        if not OCR_AVAILABLE:
            errors.append("OCR not available. Install pytesseract and PIL.")
            return ParsedDocument(
                structure=DocumentStructure(elements=[], reading_order=[]),
                metadata={},
                file_path=file_path,
                file_type="image",
                parser_used="PositionalImageParser",
                errors=errors
            )
        
        try:
            image = Image.open(file_path)
            
            metadata = {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height
            }
            
            page_dimensions[0] = (image.width, image.height)
            
            # Get detailed OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Group OCR data into text blocks
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    confidence = float(ocr_data['conf'][i])
                    if confidence > 30:  # Filter low-confidence detections
                        bbox = BoundingBox(
                            x0=float(ocr_data['left'][i]),
                            y0=float(ocr_data['top'][i]),
                            x1=float(ocr_data['left'][i] + ocr_data['width'][i]),
                            y1=float(ocr_data['top'][i] + ocr_data['height'][i]),
                            page=0
                        )
                        
                        element = DocumentElement(
                            content=text,
                            element_type=ElementType.TEXT,
                            bbox=bbox,
                            confidence=confidence / 100.0,
                            metadata={'ocr_level': ocr_data['level'][i]}
                        )
                        elements.append(element)
            
        except Exception as e:
            errors.append(f"Image parsing error: {str(e)}")
        
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
            file_type="image",
            parser_used="PositionalImageParser",
            errors=errors if errors else None
        )