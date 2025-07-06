from enum import Enum


class ElementType(Enum):
    """Types of document elements"""
    TEXT = "text"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    FOOTER = "footer"
    HEADER = "header"
    SIDEBAR = "sidebar"
    CAPTION = "caption"
    FORMULA = "formula"
    ANNOTATION = "annotation"