import csv
import logging
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Optional


from ..types import BoundingBox, DocumentElement, DocumentStructure, ElementType, ParsedDocument, CSVCell, CSVColumn, CSVRow
from .base import DocumentParser

class PositionalCSVParser(DocumentParser):
    """Enhanced CSV parser that preserves spatial information and table structure"""
    
    def __init__(self, cell_width: float = 100.0, cell_height: float = 20.0):
        """
        Initialize the CSV parser with spatial parameters
        
        Args:
            cell_width: Default width for each cell in points
            cell_height: Default height for each cell in points
        """
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.logger = logging.getLogger(__name__)
    
    def can_parse(self, file_path: str) -> bool:
        """Check if the file can be parsed as CSV"""
        return file_path.lower().endswith('.csv')
    
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse CSV file with spatial awareness"""
        elements = []
        metadata = {}
        errors = []
        page_dimensions = {}
        
        try:
            # First, analyze the CSV structure
            csv_structure = self._analyze_csv_structure(file_path)
            
            # Read the CSV data
            df = pd.read_csv(file_path, keep_default_na=False)
            
            # Basic metadata
            metadata = {
                'rows': len(df),
                'columns': len(df.columns),
                'file_size': Path(file_path).stat().st_size,
                'encoding': self._detect_encoding(file_path),
                'delimiter': csv_structure['delimiter'],
                'has_header': csv_structure['has_header'],
                'column_names': df.columns.tolist(),
                'data_types': {col: str(df[col].dtype) for col in df.columns}
            }
            
            # Calculate page dimensions based on table size
            total_width = len(df.columns) * self.cell_width
            total_height = (len(df) + 1) * self.cell_height  # +1 for header
            page_dimensions[0] = (total_width, total_height)
            
            # Create CSV-specific elements
            csv_cells = []
            csv_rows = []
            csv_columns = []
            
            # Process header row
            if csv_structure['has_header']:
                header_row = self._create_header_row(df.columns, 0)
                csv_rows.append(header_row)
                elements.extend(self._row_to_elements(header_row))
            
            # Process data rows
            for idx, (_, row) in enumerate(df.iterrows()):
                row_index = idx + (1 if csv_structure['has_header'] else 0)
                csv_row = self._create_data_row(row, row_index, df.columns)
                csv_rows.append(csv_row)
                elements.extend(self._row_to_elements(csv_row))
                csv_cells.extend(csv_row.cells)
            
            # Create column metadata
            csv_columns = self._create_column_metadata(df, csv_cells)
            
            # Create table-level element
            table_bbox = BoundingBox(
                x0=0, y0=0, 
                x1=total_width, 
                y1=total_height,
                page=0
            )
            
            table_element = DocumentElement(
                content=self._format_table_content(df),
                element_type=ElementType.TABLE,
                bbox=table_bbox,
                metadata={
                    'csv_structure': csv_structure,
                    'columns': [col.__dict__ for col in csv_columns],
                    'rows': len(csv_rows),
                    'cells': len(csv_cells),
                    'data_quality': self._assess_data_quality(csv_cells),
                    'spatial_analysis': self._analyze_spatial_patterns(csv_cells)
                }
            )
            elements.append(table_element)
            
            # Add summary statistics element
            stats_element = self._create_statistics_element(df, csv_columns)
            elements.append(stats_element)
            
        except Exception as e:
            errors.append(f"CSV parsing error: {str(e)}")
            self.logger.error(f"Error parsing CSV {file_path}: {str(e)}")
        
        # Determine reading order (table structure)
        reading_order = self._determine_csv_reading_order(elements)
        
        structure = DocumentStructure(
            elements=elements,
            reading_order=reading_order,
            page_dimensions=page_dimensions
        )
        
        return ParsedDocument(
            structure=structure,
            metadata=metadata,
            file_path=file_path,
            file_type="csv",
            parser_used="PositionalCSVParser",
            errors=errors if errors else None
        )
    
    def _analyze_csv_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze the structure of the CSV file"""
        structure = {
            'delimiter': ',',
            'quotechar': '"',
            'has_header': True,
            'line_count': 0,
            'max_columns': 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                # Use csv.Sniffer to detect format
                sample = f.read(1024)
                f.seek(0)
                
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample, delimiters=',;\t|')
                    structure['delimiter'] = dialect.delimiter
                    structure['quotechar'] = dialect.quotechar
                    structure['has_header'] = sniffer.has_header(sample)
                except:
                    pass  # Use defaults
                
                # Count lines and max columns
                reader = csv.reader(f, delimiter=structure['delimiter'])
                line_count = 0
                max_cols = 0
                
                for row in reader:
                    line_count += 1
                    max_cols = max(max_cols, len(row))
                
                structure['line_count'] = line_count
                structure['max_columns'] = max_cols
                
        except Exception as e:
            self.logger.warning(f"Could not analyze CSV structure: {str(e)}")
        
        return structure
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def _create_header_row(self, columns: List[str], row_index: int) -> CSVRow:
        """Create a header row with spatial information"""
        cells = []
        
        for col_idx, col_name in enumerate(columns):
            bbox = BoundingBox(
                x0=col_idx * self.cell_width,
                y0=row_index * self.cell_height,
                x1=(col_idx + 1) * self.cell_width,
                y1=(row_index + 1) * self.cell_height,
                page=0
            )
            
            cell = CSVCell(
                value=col_name,
                row=row_index,
                column=col_idx,
                column_name=col_name,
                bbox=bbox,
                is_header=True
            )
            cells.append(cell)
        
        row_bbox = BoundingBox(
            x0=0,
            y0=row_index * self.cell_height,
            x1=len(columns) * self.cell_width,
            y1=(row_index + 1) * self.cell_height,
            page=0
        )
        
        return CSVRow(
            index=row_index,
            cells=cells,
            bbox=row_bbox,
            is_header=True
        )
    
    def _create_data_row(self, row_data: pd.Series, row_index: int, columns: List[str]) -> CSVRow:
        """Create a data row with spatial information"""
        cells = []
        
        for col_idx, col_name in enumerate(columns):
            value = row_data.iloc[col_idx] if col_idx < len(row_data) else ""
            
            bbox = BoundingBox(
                x0=col_idx * self.cell_width,
                y0=row_index * self.cell_height,
                x1=(col_idx + 1) * self.cell_width,
                y1=(row_index + 1) * self.cell_height,
                page=0
            )
            
            cell = CSVCell(
                value=str(value),
                row=row_index,
                column=col_idx,
                column_name=col_name,
                bbox=bbox
            )
            cells.append(cell)
        
        row_bbox = BoundingBox(
            x0=0,
            y0=row_index * self.cell_height,
            x1=len(columns) * self.cell_width,
            y1=(row_index + 1) * self.cell_height,
            page=0
        )
        
        return CSVRow(
            index=row_index,
            cells=cells,
            bbox=row_bbox
        )
    
    def _create_column_metadata(self, df: pd.DataFrame, cells: List[CSVCell]) -> List[CSVColumn]:
        """Create column metadata with spatial information"""
        columns = []
        
        for col_idx, col_name in enumerate(df.columns):
            # Get all cells for this column
            column_cells = [cell for cell in cells if cell.column == col_idx and not cell.is_header]
            
            # Calculate statistics
            data_types = list(set(cell.data_type for cell in column_cells))
            unique_values = len(set(cell.value for cell in column_cells))
            null_count = sum(1 for cell in column_cells if cell.is_empty)
            max_length = max(len(cell.value) for cell in column_cells) if column_cells else 0
            
            # Column bounding box
            bbox = BoundingBox(
                x0=col_idx * self.cell_width,
                y0=0,
                x1=(col_idx + 1) * self.cell_width,
                y1=len(df) * self.cell_height,
                page=0
            )
            
            # Statistical summary
            numeric_values = []
            for cell in column_cells:
                if cell.is_numeric:
                    try:
                        numeric_values.append(float(cell.value))
                    except ValueError:
                        pass
            
            statistical_summary = {}
            if numeric_values:
                statistical_summary = {
                    'count': len(numeric_values),
                    'mean': sum(numeric_values) / len(numeric_values),
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'std': pd.Series(numeric_values).std()
                }
            
            column = CSVColumn(
                name=col_name,
                index=col_idx,
                data_types=data_types,
                unique_values=unique_values,
                null_count=null_count,
                max_length=max_length,
                bbox=bbox,
                statistical_summary=statistical_summary
            )
            columns.append(column)
        
        return columns
    
    def _row_to_elements(self, csv_row: CSVRow) -> List[DocumentElement]:
        """Convert CSV row to document elements"""
        elements = []
        
        # Row-level element
        row_element = DocumentElement(
            content="\t".join(cell.value for cell in csv_row.cells),
            element_type=ElementType.TEXT if not csv_row.is_header else ElementType.HEADING,
            bbox=csv_row.bbox,
            metadata={
                'row_index': csv_row.index,
                'is_header': csv_row.is_header,
                'completeness': csv_row.completeness,
                'cell_count': len(csv_row.cells)
            }
        )
        elements.append(row_element)
        
        # Individual cell elements
        for cell in csv_row.cells:
            cell_element = DocumentElement(
                content=cell.value,
                element_type=ElementType.TEXT,
                bbox=cell.bbox,
                metadata={
                    'row': cell.row,
                    'column': cell.column,
                    'column_name': cell.column_name,
                    'data_type': cell.data_type,
                    'is_header': cell.is_header,
                    'is_numeric': cell.is_numeric,
                    'is_empty': cell.is_empty
                }
            )
            elements.append(cell_element)
        
        return elements
    
    def _format_table_content(self, df: pd.DataFrame) -> str:
        """Format the entire table as text content"""
        lines = []
        
        # Header
        lines.append("\t".join(df.columns))
        
        # Data rows
        for _, row in df.iterrows():
            lines.append("\t".join(str(val) for val in row.values))
        
        return "\n".join(lines)
    
    def _assess_data_quality(self, cells: List[CSVCell]) -> Dict[str, Any]:
        """Assess the quality of the CSV data"""
        total_cells = len(cells)
        empty_cells = sum(1 for cell in cells if cell.is_empty)
        numeric_cells = sum(1 for cell in cells if cell.is_numeric)
        
        return {
            'total_cells': total_cells,
            'empty_cells': empty_cells,
            'empty_percentage': empty_cells / total_cells if total_cells > 0 else 0,
            'numeric_cells': numeric_cells,
            'numeric_percentage': numeric_cells / total_cells if total_cells > 0 else 0,
            'completeness_score': 1 - (empty_cells / total_cells) if total_cells > 0 else 0
        }
    
    def _analyze_spatial_patterns(self, cells: List[CSVCell]) -> Dict[str, Any]:
        """Analyze spatial patterns in the CSV data"""
        patterns = {
            'data_distribution': {},
            'column_patterns': {},
            'row_patterns': {}
        }
        
        # Group cells by column
        columns = {}
        for cell in cells:
            if cell.column not in columns:
                columns[cell.column] = []
            columns[cell.column].append(cell)
        
        # Analyze column patterns
        for col_idx, col_cells in columns.items():
            empty_count = sum(1 for cell in col_cells if cell.is_empty)
            patterns['column_patterns'][col_idx] = {
                'empty_ratio': empty_count / len(col_cells) if col_cells else 0,
                'data_type_distribution': {}
            }
            
            # Data type distribution
            type_counts = {}
            for cell in col_cells:
                type_counts[cell.data_type] = type_counts.get(cell.data_type, 0) + 1
            patterns['column_patterns'][col_idx]['data_type_distribution'] = type_counts
        
        # Group cells by row
        rows = {}
        for cell in cells:
            if cell.row not in rows:
                rows[cell.row] = []
            rows[cell.row].append(cell)
        
        # Analyze row patterns
        for row_idx, row_cells in rows.items():
            empty_count = sum(1 for cell in row_cells if cell.is_empty)
            patterns['row_patterns'][row_idx] = {
                'empty_ratio': empty_count / len(row_cells) if row_cells else 0,
                'completeness': 1 - (empty_count / len(row_cells)) if row_cells else 0
            }
        
        return patterns
    
    def _create_statistics_element(self, df: pd.DataFrame, columns: List[CSVColumn]) -> DocumentElement:
        """Create an element containing statistical summary"""
        stats_content = []
        stats_content.append("CSV Statistics:")
        stats_content.append(f"Dimensions: {len(df)} rows × {len(df.columns)} columns")
        stats_content.append("")
        
        for col in columns:
            stats_content.append(f"Column: {col.name}")
            stats_content.append(f"  Data types: {', '.join(col.data_types)}")
            stats_content.append(f"  Unique values: {col.unique_values}")
            stats_content.append(f"  Null count: {col.null_count}")
            stats_content.append(f"  Max length: {col.max_length}")
            
            if col.statistical_summary:
                stats_content.append(f"  Statistics: {col.statistical_summary}")
            stats_content.append("")
        
        # Position at the bottom of the table
        bbox = BoundingBox(
            x0=0,
            y0=len(df) * self.cell_height + 20,
            x1=len(df.columns) * self.cell_width,
            y1=len(df) * self.cell_height + 200,
            page=0
        )
        
        return DocumentElement(
            content="\n".join(stats_content),
            element_type=ElementType.ANNOTATION,
            bbox=bbox,
            metadata={
                'type': 'statistics',
                'columns': [col.__dict__ for col in columns]
            }
        )
    
    def _determine_csv_reading_order(self, elements: List[DocumentElement]) -> List[str]:
        """Determine reading order for CSV elements (row by row, left to right)"""
        # Filter out non-content elements
        content_elements = [e for e in elements if e.element_type in [
            ElementType.TEXT, ElementType.HEADING, ElementType.TABLE
        ]]
        
        # Sort by row (Y coordinate), then by column (X coordinate)
        content_elements.sort(key=lambda e: (e.bbox.y0, e.bbox.x0))
        
        return [e.element_id for e in content_elements]
    
    def get_cell_at_position(self, result: ParsedDocument, x: float, y: float) -> Optional[Dict[str, Any]]:
        """Get the cell at a specific position"""
        for element in result.structure.elements:
            if (element.bbox.x0 <= x <= element.bbox.x1 and 
                element.bbox.y0 <= y <= element.bbox.y1 and
                'column' in element.metadata):
                return {
                    'content': element.content,
                    'row': element.metadata['row'],
                    'column': element.metadata['column'],
                    'column_name': element.metadata['column_name'],
                    'data_type': element.metadata['data_type'],
                    'bbox': element.bbox
                }
        return None
    
    def get_column_data(self, result: ParsedDocument, column_name: str) -> List[Dict[str, Any]]:
        """Get all data for a specific column"""
        column_data = []
        
        for element in result.structure.elements:
            if (element.metadata.get('column_name') == column_name and
                not element.metadata.get('is_header', False)):
                column_data.append({
                    'content': element.content,
                    'row': element.metadata['row'],
                    'data_type': element.metadata['data_type'],
                    'bbox': element.bbox
                })
        
        # Sort by row
        column_data.sort(key=lambda x: x['row'])
        return column_data
    
    def get_row_data(self, result: ParsedDocument, row_index: int) -> List[Dict[str, Any]]:
        """Get all data for a specific row"""
        row_data = []
        
        for element in result.structure.elements:
            if element.metadata.get('row') == row_index:
                row_data.append({
                    'content': element.content,
                    'column': element.metadata['column'],
                    'column_name': element.metadata['column_name'],
                    'data_type': element.metadata['data_type'],
                    'bbox': element.bbox
                })
        
        # Sort by column
        row_data.sort(key=lambda x: x['column'])
        return row_data
    
    def find_cells_by_content(self, result: ParsedDocument, search_term: str, 
                             case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Find cells containing specific content"""
        matches = []
        
        for element in result.structure.elements:
            if 'column' in element.metadata:  # It's a cell
                content = element.content
                search = search_term
                
                if not case_sensitive:
                    content = content.lower()
                    search = search.lower()
                
                if search in content:
                    matches.append({
                        'content': element.content,
                        'row': element.metadata['row'],
                        'column': element.metadata['column'],
                        'column_name': element.metadata['column_name'],
                        'bbox': element.bbox
                    })
        
        return matches
    
    def get_data_range(self, result: ParsedDocument, start_row: int, end_row: int, 
                      start_col: int, end_col: int) -> List[List[str]]:
        """Get data from a specific range of cells"""
        range_data = {}
        
        for element in result.structure.elements:
            if ('row' in element.metadata and 'column' in element.metadata):
                row = element.metadata['row']
                col = element.metadata['column']
                
                if (start_row <= row <= end_row and start_col <= col <= end_col):
                    if row not in range_data:
                        range_data[row] = {}
                    range_data[row][col] = element.content
        
        # Convert to 2D array
        result_array = []
        for row in range(start_row, end_row + 1):
            row_data = []
            for col in range(start_col, end_col + 1):
                cell_value = range_data.get(row, {}).get(col, "")
                row_data.append(cell_value)
            result_array.append(row_data)
        
        return result_array