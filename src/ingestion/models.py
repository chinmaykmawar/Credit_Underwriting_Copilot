from dataclasses import dataclass, field

@dataclass
class Page():
    page_number:int
    text:str

@dataclass
class DocumentMetadata():
    company_name: str
    document_type: str
    source_category: str
    source_file: str
    financial_year: int

@dataclass
class Document():
    metadata: DocumentMetadata
    pages: dict[int,str] = field(default_factory=dict)