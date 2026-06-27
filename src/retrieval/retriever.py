from src.chunking.models import Chunk
from src.retrieval.VectorStore import VectorStore
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    distance: float
    chunk: Chunk

class Retriever:

    def __init__(self,vector_store: VectorStore,chunks: list[Chunk]):
        self.vector_store = vector_store
        self.chunks = chunks

    def retrieve(self,query_embedding,top_k: int = 5) -> list[RetrievalResult]:
        distances, indices = self.vector_store.search(query_embedding=query_embedding,top_k=top_k)
        results = [RetrievalResult(d, self.chunks[idx]) for d,idx in zip(distances,indices) if idx!=-1]      
        return results
        