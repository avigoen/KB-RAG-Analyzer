
import re
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

from .base import DocumentParser
from ..layout_analyser import LayoutAnalyzer
from ..types import BoundingBox, DocumentElement, DocumentStructure, ElementType, ParsedDocument


class PositionalXMLParser(DocumentParser):
    """Enhanced XML parser that preserves hierarchical structure and spatial relationships"""
    
    def __init__(self):
        self.namespace_map = {}
        self.element_counter = 0
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.xml', '.xsd', '.xsl', '.xslt', '.svg', '.rss', '.atom'))
    
    def parse(self, file_path: str) -> ParsedDocument:
        elements = []
        metadata = {}
        errors = []
        page_dimensions = {}
        
        try:
            # Parse XML with namespace handling
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract namespace information
            self.namespace_map = self._extract_namespaces(root)
            
            # Get file statistics
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            metadata = {
                'root_tag': root.tag,
                'namespaces': self.namespace_map,
                'total_elements': len(list(root.iter())),
                'file_size': len(content),
                'encoding': 'utf-8',
                'xml_version': '1.0',  # Default assumption
                'depth': self._calculate_tree_depth(root),
                'unique_tags': len(set(elem.tag for elem in root.iter())),
                'attributes_count': sum(len(elem.attrib) for elem in root.iter()),
                'text_nodes': sum(1 for elem in root.iter() if elem.text and elem.text.strip()),
            }
            
            # Extract XML declaration if present
            xml_declaration = self._extract_xml_declaration(content)
            if xml_declaration:
                metadata.update(xml_declaration)
            
            # Process elements with hierarchical positioning
            y_position = 0
            x_indent = 0
            self.element_counter = 0
            
            # Process root element and its children
            root_elements = self._process_element_hierarchy(root, x_indent, y_position, 0)
            elements.extend(root_elements)
            
            # Calculate page dimensions based on content
            if elements:
                max_x = max(elem.bbox.x1 for elem in elements)
                max_y = max(elem.bbox.y1 for elem in elements)
                page_dimensions[0] = (max_x, max_y)
            else:
                page_dimensions[0] = (800, 600)  # Default dimensions
            
            # Special handling for specific XML types
            xml_type = self._detect_xml_type(root)
            metadata['xml_type'] = xml_type
            
            if xml_type == 'svg':
                elements = self._enhance_svg_elements(elements, root)
            elif xml_type == 'rss' or xml_type == 'atom':
                elements = self._enhance_feed_elements(elements, root)
            elif xml_type == 'xsd':
                elements = self._enhance_schema_elements(elements, root)
                
        except ET.ParseError as e:
            errors.append(f"XML parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"General parsing error: {str(e)}")
        
        # Determine reading order based on XML hierarchy
        reading_order = self._determine_xml_reading_order(elements)
        
        # Build hierarchical relationships
        relationships = self._build_xml_relationships(elements)
        
        structure = DocumentStructure(
            elements=elements,
            reading_order=reading_order,
            relationships=relationships,
            page_dimensions=page_dimensions
        )
        
        return ParsedDocument(
            structure=structure,
            metadata=metadata,
            file_path=file_path,
            file_type="xml",
            parser_used="PositionalXMLParser",
            errors=errors if errors else None
        )
    
    def _extract_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """Extract namespace declarations from XML"""
        namespaces = {}
        
        # Get namespaces from root element
        for prefix, uri in root.attrib.items():
            if prefix.startswith('xmlns'):
                if prefix == 'xmlns':
                    namespaces['default'] = uri
                else:
                    namespaces[prefix.split(':')[1]] = uri
        
        # Extract namespaces from tag names
        for elem in root.iter():
            if '}' in elem.tag:
                namespace = elem.tag.split('}')[0][1:]  # Remove { and }
                if namespace not in namespaces.values():
                    namespaces[f'ns{len(namespaces)}'] = namespace
        
        return namespaces
    
    def _extract_xml_declaration(self, content: str) -> Optional[Dict[str, str]]:
        """Extract XML declaration information"""
        declaration_pattern = r'<\?xml\s+([^?]+)\?>'
        match = re.search(declaration_pattern, content)
        
        if match:
            declaration = {}
            attrs = match.group(1)
            
            # Parse version
            version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', attrs)
            if version_match:
                declaration['xml_version'] = version_match.group(1)
            
            # Parse encoding
            encoding_match = re.search(r'encoding\s*=\s*["\']([^"\']+)["\']', attrs)
            if encoding_match:
                declaration['encoding'] = encoding_match.group(1)
            
            # Parse standalone
            standalone_match = re.search(r'standalone\s*=\s*["\']([^"\']+)["\']', attrs)
            if standalone_match:
                declaration['standalone'] = standalone_match.group(1)
            
            return declaration
        
        return None
    
    def _calculate_tree_depth(self, root: ET.Element) -> int:
        """Calculate the maximum depth of the XML tree"""
        def get_depth(elem, current_depth=0):
            if len(elem) == 0:
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in elem)
        
        return get_depth(root)
    
    def _process_element_hierarchy(self, element: ET.Element, x_indent: int, y_position: int, 
                                  depth: int) -> List[DocumentElement]:
        """Process XML element and its children with hierarchical positioning"""
        elements = []
        current_y = y_position
        
        # Determine element type based on content and structure
        element_type = self._classify_xml_element(element)
        
        # Calculate bounding box
        tag_name = self._clean_tag_name(element.tag)
        content = self._get_element_content(element)
        
        # Element dimensions based on content and depth
        element_width = max(200, len(content) * 8)  # Approximate width
        element_height = 20 + (5 * depth)  # Height increases with depth
        
        bbox = BoundingBox(
            x0=x_indent,
            y0=current_y,
            x1=x_indent + element_width,
            y1=current_y + element_height,
            page=0
        )
        
        # Create document element
        self.element_counter += 1
        doc_element = DocumentElement(
            content=content,
            element_type=element_type,
            bbox=bbox,
            style={
                'tag_name': tag_name,
                'depth': depth,
                'attributes': dict(element.attrib),
                'namespace': self._get_namespace(element.tag),
                'has_children': len(element) > 0,
                'has_text': bool(element.text and element.text.strip())
            },
            element_id=f"xml_element_{self.element_counter}",
            metadata={
                'xpath': self._get_xpath(element),
                'tag_full': element.tag,
                'tag_local': tag_name,
                'attributes': dict(element.attrib),
                'namespace_uri': self._get_namespace_uri(element.tag),
                'depth': depth,
                'child_count': len(element),
                'text_content': element.text.strip() if element.text else '',
                'tail_content': element.tail.strip() if element.tail else ''
            }
        )
        
        elements.append(doc_element)
        current_y += element_height + 5  # Add spacing
        
        # Process attributes as separate elements if they're significant
        if element.attrib:
            attr_elements = self._process_attributes(element, x_indent + 20, current_y, depth + 1)
            elements.extend(attr_elements)
            current_y += len(attr_elements) * 25
        
        # Process children recursively
        child_indent = x_indent + 30  # Indent children
        for child in element:
            child_elements = self._process_element_hierarchy(child, child_indent, current_y, depth + 1)
            elements.extend(child_elements)
            if child_elements:
                current_y = max(elem.bbox.y1 for elem in child_elements) + 5
        
        return elements
    
    def _classify_xml_element(self, element: ET.Element) -> ElementType:
        """Classify XML element based on its characteristics"""
        tag_name = self._clean_tag_name(element.tag).lower()
        
        # Check for common structural elements
        if tag_name in ['title', 'header', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'heading']:
            return ElementType.HEADING
        elif tag_name in ['p', 'paragraph', 'text', 'description', 'summary']:
            return ElementType.PARAGRAPH
        elif tag_name in ['table', 'grid', 'matrix']:
            return ElementType.TABLE
        elif tag_name in ['img', 'image', 'picture', 'graphic']:
            return ElementType.IMAGE
        elif tag_name in ['list', 'ul', 'ol', 'items']:
            return ElementType.LIST
        elif tag_name in ['annotation', 'comment', 'note']:
            return ElementType.ANNOTATION
        elif tag_name in ['footer', 'bottom']:
            return ElementType.FOOTER
        elif tag_name in ['sidebar', 'aside']:
            return ElementType.SIDEBAR
        elif tag_name in ['caption', 'label']:
            return ElementType.CAPTION
        elif tag_name in ['formula', 'equation', 'math']:
            return ElementType.FORMULA
        else:
            # Default classification based on content
            if len(element) > 0:
                return ElementType.TEXT  # Container element
            elif element.text and element.text.strip():
                return ElementType.TEXT  # Text element
            else:
                return ElementType.TEXT  # Empty or structural element
    
    def _clean_tag_name(self, tag: str) -> str:
        """Remove namespace prefix from tag name"""
        if '}' in tag:
            return tag.split('}')[1]
        return tag
    
    def _get_namespace(self, tag: str) -> Optional[str]:
        """Get namespace prefix for a tag"""
        if '}' in tag:
            namespace_uri = tag.split('}')[0][1:]
            for prefix, uri in self.namespace_map.items():
                if uri == namespace_uri:
                    return prefix
        return None
    
    def _get_namespace_uri(self, tag: str) -> Optional[str]:
        """Get namespace URI for a tag"""
        if '}' in tag:
            return tag.split('}')[0][1:]
        return None
    
    def _get_element_content(self, element: ET.Element) -> str:
        """Get meaningful content from XML element"""
        tag_name = self._clean_tag_name(element.tag)
        
        # Priority order for content
        if element.text and element.text.strip():
            return element.text.strip()
        elif element.attrib:
            # Use key attributes as content
            key_attrs = ['name', 'id', 'title', 'value', 'href', 'src']
            for attr in key_attrs:
                if attr in element.attrib:
                    return f"{tag_name}: {element.attrib[attr]}"
        
        # Fallback to tag name with child count
        if len(element) > 0:
            return f"<{tag_name}> ({len(element)} children)"
        else:
            return f"<{tag_name}>"
    
    def _get_xpath(self, element: ET.Element) -> str:
        """Generate XPath for element (simplified)"""
        # This is a simplified XPath generation
        # In a real implementation, you might want to use lxml for accurate XPath
        return f"//{self._clean_tag_name(element.tag)}"
    
    def _process_attributes(self, element: ET.Element, x_indent: int, y_position: int, 
                           depth: int) -> List[DocumentElement]:
        """Process XML attributes as separate elements"""
        attr_elements = []
        current_y = y_position
        
        for attr_name, attr_value in element.attrib.items():
            self.element_counter += 1
            
            bbox = BoundingBox(
                x0=x_indent,
                y0=current_y,
                x1=x_indent + max(200, len(f"{attr_name}={attr_value}") * 8),
                y1=current_y + 20,
                page=0
            )
            
            attr_element = DocumentElement(
                content=f"@{attr_name}={attr_value}",
                element_type=ElementType.ANNOTATION,
                bbox=bbox,
                style={
                    'attribute_name': attr_name,
                    'attribute_value': attr_value,
                    'depth': depth,
                    'is_attribute': True
                },
                element_id=f"xml_attr_{self.element_counter}",
                metadata={
                    'attribute_name': attr_name,
                    'attribute_value': attr_value,
                    'parent_tag': self._clean_tag_name(element.tag),
                    'depth': depth
                }
            )
            
            attr_elements.append(attr_element)
            current_y += 25
        
        return attr_elements
    
    def _detect_xml_type(self, root: ET.Element) -> str:
        """Detect the type of XML document"""
        root_tag = self._clean_tag_name(root.tag).lower()
        
        if root_tag == 'svg':
            return 'svg'
        elif root_tag in ['rss', 'feed']:
            return 'rss' if root_tag == 'rss' else 'atom'
        elif root_tag in ['schema', 'xsd']:
            return 'xsd'
        elif root_tag in ['stylesheet', 'transform']:
            return 'xslt'
        elif root_tag in ['html', 'xhtml']:
            return 'xhtml'
        elif root_tag in ['configuration', 'config']:
            return 'config'
        elif root_tag in ['pom', 'project']:
            return 'maven'
        elif root_tag == 'beans':
            return 'spring'
        elif 'namespace' in root.attrib and 'android' in str(root.attrib.values()):
            return 'android'
        else:
            return 'generic'
    
    def _enhance_svg_elements(self, elements: List[DocumentElement], root: ET.Element) -> List[DocumentElement]:
        """Enhance SVG elements with specific positioning"""
        enhanced_elements = []
        
        for element in elements:
            if element.style.get('tag_name') in ['rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'path']:
                # Update element type for SVG shapes
                element.element_type = ElementType.IMAGE
                
                # Try to extract actual SVG coordinates
                attrs = element.metadata.get('attributes', {})
                if 'x' in attrs and 'y' in attrs:
                    try:
                        x = float(attrs['x'])
                        y = float(attrs['y'])
                        width = float(attrs.get('width', 50))
                        height = float(attrs.get('height', 50))
                        
                        # Update bounding box with SVG coordinates
                        element.bbox = BoundingBox(x0=x, y0=y, x1=x+width, y1=y+height, page=0)
                    except (ValueError, TypeError):
                        pass
            
            enhanced_elements.append(element)
        
        return enhanced_elements
    
    def _enhance_feed_elements(self, elements: List[DocumentElement], root: ET.Element) -> List[DocumentElement]:
        """Enhance RSS/Atom feed elements"""
        enhanced_elements = []
        
        for element in elements:
            tag_name = element.style.get('tag_name', '').lower()
            
            if tag_name in ['title', 'name']:
                element.element_type = ElementType.HEADING
            elif tag_name in ['description', 'summary', 'content']:
                element.element_type = ElementType.PARAGRAPH
            elif tag_name in ['link', 'url']:
                element.element_type = ElementType.ANNOTATION
            
            enhanced_elements.append(element)
        
        return enhanced_elements
    
    def _enhance_schema_elements(self, elements: List[DocumentElement], root: ET.Element) -> List[DocumentElement]:
        """Enhance XSD schema elements"""
        enhanced_elements = []
        
        for element in elements:
            tag_name = element.style.get('tag_name', '').lower()
            
            if tag_name in ['element', 'attribute']:
                element.element_type = ElementType.ANNOTATION
            elif tag_name in ['complextype', 'simpletype']:
                element.element_type = ElementType.HEADING
            elif tag_name in ['documentation', 'annotation']:
                element.element_type = ElementType.CAPTION
            
            enhanced_elements.append(element)
        
        return enhanced_elements
    
    def _determine_xml_reading_order(self, elements: List[DocumentElement]) -> List[str]:
        """Determine reading order based on XML hierarchy and depth"""
        # Sort by depth first, then by position
        sorted_elements = sorted(elements, key=lambda e: (e.style.get('depth', 0), e.bbox.y0, e.bbox.x0))
        return [elem.element_id for elem in sorted_elements]
    
    def _build_xml_relationships(self, elements: List[DocumentElement]) -> Dict[str, List[str]]:
        """Build parent-child relationships based on XML hierarchy"""
        relationships = {}
        
        # Group elements by depth
        depth_groups = {}
        for element in elements:
            depth = element.style.get('depth', 0)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(element)
        
        # Build relationships between consecutive depths
        for depth in sorted(depth_groups.keys()):
            if depth + 1 in depth_groups:
                parent_elements = depth_groups[depth]
                child_elements = depth_groups[depth + 1]
                
                for parent in parent_elements:
                    # Find children that are spatially close and at the next depth level
                    children = []
                    for child in child_elements:
                        # Check if child is positioned after parent and within reasonable bounds
                        if (child.bbox.y0 > parent.bbox.y0 and 
                            child.bbox.x0 > parent.bbox.x0 and
                            child.bbox.y0 < parent.bbox.y1 + 200):  # Reasonable proximity
                            children.append(child.element_id)
                    
                    if children:
                        relationships[parent.element_id] = children
        
        return relationships
    
    def get_elements_by_xpath(self, result: ParsedDocument, xpath_pattern: str) -> List[DocumentElement]:
        """Get elements matching XPath-like pattern"""
        matching_elements = []
        
        # Simple XPath pattern matching (basic implementation)
        if xpath_pattern.startswith('//'):
            tag_name = xpath_pattern[2:]
            for element in result.structure.elements:
                if element.style.get('tag_name') == tag_name:
                    matching_elements.append(element)
        
        return matching_elements
    
    def get_elements_by_namespace(self, result: ParsedDocument, namespace: str) -> List[DocumentElement]:
        """Get elements from specific namespace"""
        matching_elements = []
        
        for element in result.structure.elements:
            if element.style.get('namespace') == namespace:
                matching_elements.append(element)
        
        return matching_elements
    
    def get_elements_by_attribute(self, result: ParsedDocument, attr_name: str, 
                                 attr_value: Optional[str] = None) -> List[DocumentElement]:
        """Get elements with specific attribute"""
        matching_elements = []
        
        for element in result.structure.elements:
            attributes = element.metadata.get('attributes', {})
            if attr_name in attributes:
                if attr_value is None or attributes[attr_name] == attr_value:
                    matching_elements.append(element)
        
        return matching_elements
    
    def extract_xml_schema(self, result: ParsedDocument) -> Dict[str, Any]:
        """Extract XML schema information"""
        schema = {
            'elements': {},
            'attributes': {},
            'namespaces': result.metadata.get('namespaces', {}),
            'hierarchy': {}
        }
        
        # Collect element information
        for element in result.structure.elements:
            tag_name = element.style.get('tag_name')
            if tag_name:
                if tag_name not in schema['elements']:
                    schema['elements'][tag_name] = {
                        'count': 0,
                        'attributes': set(),
                        'children': set(),
                        'namespaces': set()
                    }
                
                schema['elements'][tag_name]['count'] += 1
                
                # Collect attributes
                for attr in element.metadata.get('attributes', {}):
                    schema['elements'][tag_name]['attributes'].add(attr)
                    schema['attributes'][attr] = schema['attributes'].get(attr, 0) + 1
                
                # Collect namespace
                ns = element.style.get('namespace')
                if ns:
                    schema['elements'][tag_name]['namespaces'].add(ns)
        
        # Build hierarchy
        for parent_id, children_ids in result.structure.relationships.items():
            parent_element = next((e for e in result.structure.elements if e.element_id == parent_id), None)
            if parent_element:
                parent_tag = parent_element.style.get('tag_name')
                if parent_tag:
                    for child_id in children_ids:
                        child_element = next((e for e in result.structure.elements if e.element_id == child_id), None)
                        if child_element:
                            child_tag = child_element.style.get('tag_name')
                            if child_tag:
                                schema['elements'][parent_tag]['children'].add(child_tag)
        
        # Convert sets to lists for JSON serialization
        for tag_info in schema['elements'].values():
            tag_info['attributes'] = list(tag_info['attributes'])
            tag_info['children'] = list(tag_info['children'])
            tag_info['namespaces'] = list(tag_info['namespaces'])
        
        return schema