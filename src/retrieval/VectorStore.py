from pathlib import Path
import faiss
from faiss import Index
import numpy as np
from src.chunking.models import Chunk
from collections import defaultdict


import logging
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, base_folder=r"data\processed"):
        self.embedding_dimension = None
        self.base_index_folder=Path(base_folder)

    def create_index_file(self,chunks: list[Chunk]) -> bool:
        logger.info(f'VS.CreateIndexFIle<>Entering')
        company_year_embedding=defaultdict(lambda: defaultdict(list))
        for chunk in chunks:
            chunk:Chunk
            if chunk.embedding is None:
                raise ValueError("Chunk embedding is None.")
            c_name = chunk.metadata.company_name
            f_year = chunk.metadata.financial_year
            company_year_embedding[c_name][f_year].append(chunk.embedding)
            
        for company,year_embedding in company_year_embedding.items():
            for year, embeddings in year_embedding.items():
                embedding_matrix = np.array(embeddings,dtype=np.float32)
                faiss.normalize_L2(embedding_matrix)
                if self.embedding_dimension==None:
                    self.embedding_dimension = embedding_matrix.shape[1]        
                index_filePath = self._get_isolated_paths(company,year)
                index = faiss.IndexFlatIP(self.embedding_dimension)
                index.add(embedding_matrix)
                index_filePath.unlink(missing_ok=True)
                faiss.write_index(index,str(index_filePath))
        logger.info(f'VS.CreateIndexFIle<>Exiting')
        return True

    def _load_index(self,company_name: str, financial_year: int) -> Index:
        index_file = self._get_isolated_paths(company_name, financial_year)
        if not (index_file.exists() and index_file.is_file()):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        index = faiss.read_index(str(index_file))
        return index

    def search(self,company_name: str, financial_year: int, query_embedding: np.ndarray,top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        index= self._load_index(company_name, financial_year)
        query_matrix = np.array([query_embedding],dtype=np.float32)
        faiss.normalize_L2(query_matrix)
        distances, indices = index.search(query_matrix,top_k)
        return distances[0], indices[0]
    
    def _get_isolated_paths(self, company_name: str, fiscal_year: int) -> Path:
        sanitized_company = company_name.strip().lower().replace(" ", "_")
        folder_path = self.base_index_folder.joinpath(sanitized_company,str(fiscal_year))
        folder_path.mkdir(parents=True,exist_ok=True)
        index_path = folder_path.joinpath("faiss.index")
        return index_path