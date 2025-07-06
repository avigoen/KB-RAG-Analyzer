from typing import List

from .types.document_element import DocumentElement


class LayoutAnalyzer:
    """Analyzes document layout and determines reading order"""
    
    @staticmethod
    def determine_reading_order(elements: List[DocumentElement]) -> List[str]:
        """Determine reading order based on spatial layout"""
        # Group elements by page
        pages = {}
        for element in elements:
            page = element.bbox.page
            if page not in pages:
                pages[page] = []
            pages[page].append(element)
        
        reading_order = []
        
        for page_num in sorted(pages.keys()):
            page_elements = pages[page_num]
            # Sort by Y coordinate (top to bottom), then X coordinate (left to right)
            page_elements.sort(key=lambda e: (e.bbox.y0, e.bbox.x0))
            
            # Group elements into lines/rows
            lines = LayoutAnalyzer._group_into_lines(page_elements)
            
            for line in lines:
                # Sort each line left to right
                line.sort(key=lambda e: e.bbox.x0)
                reading_order.extend([e.element_id for e in line])
        
        return reading_order
    
    @staticmethod
    def _group_into_lines(elements: List[DocumentElement], tolerance: float = 5.0) -> List[List[DocumentElement]]:
        """Group elements into horizontal lines"""
        if not elements:
            return []
        
        lines = []
        current_line = [elements[0]]
        current_y = elements[0].bbox.center_y
        
        for element in elements[1:]:
            if abs(element.bbox.center_y - current_y) <= tolerance:
                current_line.append(element)
            else:
                lines.append(current_line)
                current_line = [element]
                current_y = element.bbox.center_y
        
        lines.append(current_line)
        return lines
    
    @staticmethod
    def detect_columns(elements: List[DocumentElement]) -> List[List[DocumentElement]]:
        """Detect column layout"""
        if not elements:
            return []
        
        # Find column boundaries by analyzing X coordinates
        x_coords = []
        for element in elements:
            x_coords.extend([element.bbox.x0, element.bbox.x1])
        
        # Use histogram to find column boundaries
        hist, bin_edges = np.histogram(x_coords, bins=50)
        
        # Find peaks (potential column boundaries)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                peaks.append(bin_edges[i])
        
        if len(peaks) < 2:
            return [elements]  # Single column
        
        # Assign elements to columns
        columns = [[] for _ in range(len(peaks) + 1)]
        for element in elements:
            column_idx = 0
            for i, peak in enumerate(peaks):
                if element.bbox.center_x > peak:
                    column_idx = i + 1
            columns[column_idx].append(element)
        
        return [col for col in columns if col]  # Remove empty columns