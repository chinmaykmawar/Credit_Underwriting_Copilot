from src.chunking.chunker import Chunker, Chunk
from src.retrieval.VectorStore import VectorStore
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    distance: float
    chunk: Chunk

class Retriever:

    def __init__(self,vector_store: VectorStore,chunker:Chunker):
        self.vector_store = vector_store
        self.chunker=chunker

    def retrieve(self,company_name:str, financial_year:int,query_embedding, top_k: int = 5) -> list[RetrievalResult]:
        distances, indices = self.vector_store.search(company_name, financial_year,query_embedding,top_k)
        isolated_chunks = self.chunker.load_chunks(company_name, financial_year)
        if not isolated_chunks:
            print(f"⚠️ No chunk metadata found for {company_name} ({financial_year}).")
            return []
        else:
            results = [RetrievalResult(d, isolated_chunks[idx]) for d,idx in zip(distances,indices) if idx!=-1]      
            return results
        