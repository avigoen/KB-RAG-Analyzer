from abc import ABC, abstractmethod

from ..types.parsed_document import ParsedDocument


class DocumentParser(ABC):
    """Abstract base class for document parsers"""
    
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file"""
        pass
    
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse the document and return structured content"""
        pass