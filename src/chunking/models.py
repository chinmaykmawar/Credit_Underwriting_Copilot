import numpy as np
from dataclasses import dataclass, field
from src.ingestion.models import DocumentMetadata

@dataclass
class Chunk():
    text : str
    start_page: int
    end_page:int
    metadata: DocumentMetadata
    embedding : np.ndarray | None = None
    