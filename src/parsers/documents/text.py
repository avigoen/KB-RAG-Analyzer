import re
from typing import Any, Dict, List

from .base import DocumentParser
from ..layout_analyser import LayoutAnalyzer
from ..types import BoundingBox, DocumentElement, DocumentStructure, ElementType, ParsedDocument


class PositionalTextParser(DocumentParser):
    """Enhanced text parser that analyzes structure and positioning"""
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.txt', '.md', '.rst'))
    
    def parse(self, file_path: str) -> ParsedDocument:
        elements = []
        metadata = {}
        errors = []
        page_dimensions = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            metadata = {
                'total_lines': len(lines),
                'total_characters': len(content),
                'encoding': 'utf-8',
                'line_endings': self._detect_line_endings(content)
            }
            
            # Analyze text structure
            structure_analysis = self._analyze_text_structure(lines)
            metadata.update(structure_analysis)
            
            y_position = 0
            max_line_length = 0
            
            for line_num, line in enumerate(lines):
                if line.strip():  # Skip empty lines for positioning
                    max_line_length = max(max_line_length, len(line))
                    
                    # Classify line type
                    element_type = self._classify_text_line(line, line_num, lines)
                    
                    # Calculate position based on line number and content
                    bbox = BoundingBox(
                        x0=0,
                        y0=y_position,
                        x1=len(line) * 8,  # Approximate character width
                        y1=y_position + 20,  # Line height
                        page=0
                    )
                    
                    # Extract line-level metadata
                    line_metadata = {
                        'line_number': line_num + 1,
                        'indent_level': len(line) - len(line.lstrip()),
                        'word_count': len(line.split()),
                        'character_count': len(line),
                        'starts_with_number': line.strip() and line.strip()[0].isdigit(),
                        'starts_with_bullet': line.strip().startswith(('•', '-', '*', '+')),
                        'is_all_caps': line.strip().isupper() if line.strip() else False
                    }
                    
                    element = DocumentElement(
                        content=line,
                        element_type=element_type,
                        bbox=bbox,
                        metadata=line_metadata
                    )
                    elements.append(element)
                
                y_position += 20
            
            page_dimensions[0] = (max_line_length * 8, y_position)
            
            # Detect and extract special structures
            special_elements = self._extract_special_structures(content)
            elements.extend(special_elements)
            
        except Exception as e:
            errors.append(f"Text parsing error: {str(e)}")
        
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
            file_type="text",
            parser_used="PositionalTextParser",
            errors=errors if errors else None
        )
    
    def _detect_line_endings(self, content: str) -> str:
        """Detect line ending style"""
        if '\r\n' in content:
            return 'CRLF'
        elif '\r' in content:
            return 'CR'
        elif '\n' in content:
            return 'LF'
        else:
            return 'None'
    
    def _analyze_text_structure(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze overall text structure"""
        analysis = {
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'indented_lines': sum(1 for line in lines if line.startswith(' ') or line.startswith('\t')),
            'bulleted_lines': sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '+'))),
            'numbered_lines': sum(1 for line in lines if line.strip() and line.strip()[0].isdigit()),
            'all_caps_lines': sum(1 for line in lines if line.strip().isupper() and line.strip()),
            'max_line_length': max(len(line) for line in lines) if lines else 0,
            'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
        }
        
        # Detect if it's structured text (markdown, etc.)
        markdown_indicators = sum(1 for line in lines if line.strip().startswith('#'))
        rst_indicators = sum(1 for line in lines if '=' in line or '-' in line)
        
        if markdown_indicators > 0:
            analysis['likely_format'] = 'markdown'
        elif rst_indicators > 5:
            analysis['likely_format'] = 'restructuredtext'
        else:
            analysis['likely_format'] = 'plain_text'
        
        return analysis
    
    def _classify_text_line(self, line: str, line_num: int, all_lines: List[str]) -> ElementType:
        """Classify a text line based on its content and context"""
        stripped = line.strip()
        
        if not stripped:
            return ElementType.TEXT
        
        # Check for markdown headings
        if stripped.startswith('#'):
            return ElementType.HEADING
        
        # Check for underlined headings (RST style)
        if line_num < len(all_lines) - 1:
            next_line = all_lines[line_num + 1].strip()
            if next_line and len(set(next_line)) == 1 and next_line[0] in '=-~^':
                return ElementType.HEADING
        
        # Check for list items
        if stripped.startswith(('•', '-', '*', '+')) or (stripped and stripped[0].isdigit() and '.' in stripped[:5]):
            return ElementType.LIST
        
        # Check for all caps (potential heading)
        if stripped.isupper() and len(stripped) < 100:
            return ElementType.HEADING
        
        # Check for code blocks or special formatting
        if stripped.startswith('```') or stripped.startswith('    '):
            return ElementType.ANNOTATION
        
        return ElementType.PARAGRAPH
    
    def _extract_special_structures(self, content: str) -> List[DocumentElement]:
        """Extract special structures like code blocks, tables, etc."""
        elements = []
        
        # Extract code blocks
        code_blocks = re.finditer(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
        for match in code_blocks:
            start_pos = match.start()
            lines_before = content[:start_pos].count('\n')
            
            bbox = BoundingBox(
                x0=0,
                y0=lines_before * 20,
                x1=max(len(line) for line in match.group(2).split('\n')) * 8,
                y1=(lines_before + match.group(2).count('\n')) * 20,
                page=0
            )
            
            element = DocumentElement(
                content=match.group(2),
                element_type=ElementType.ANNOTATION,
                bbox=bbox,
                metadata={
                    'structure_type': 'code_block',
                    'language': match.group(1) if match.group(1) else 'unknown'
                }
            )
            elements.append(element)
        
        # Extract simple tables (pipe-separated)
        table_pattern = r'(\|.*\|(?:\n\|.*\|)*)'
        tables = re.finditer(table_pattern, content, re.MULTILINE)
        for match in tables:
            start_pos = match.start()
            lines_before = content[:start_pos].count('\n')
            table_lines = match.group(1).split('\n')
            
            bbox = BoundingBox(
                x0=0,
                y0=lines_before * 20,
                x1=max(len(line) for line in table_lines) * 8,
                y1=(lines_before + len(table_lines)) * 20,
                page=0
            )
            
            element = DocumentElement(
                content=match.group(1),
                element_type=ElementType.TABLE,
                bbox=bbox,
                metadata={
                    'structure_type': 'markdown_table',
                    'rows': len(table_lines),
                    'columns': table_lines[0].count('|') - 1 if table_lines else 0
                }
            )
            elements.append(element)
        
        return elements