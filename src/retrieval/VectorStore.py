from pathlib import Path
import faiss
import numpy as np
from src.chunking.models import Chunk

class VectorStore:
    def __init__(self):
        self.index = None
        self.embedding_dimension = None

    def create_index(self,chunks: list[Chunk]) -> None:
        embeddings = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("Chunk embedding is None.")
            embeddings.append(chunk.embedding)
        
        embedding_matrix = np.array(embeddings,dtype=np.float32)
        self.embedding_dimension = embedding_matrix.shape[1]
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.index.add(embedding_matrix)

    def save_index(self,file_path: str) -> None:
        if self.index is None:
            raise ValueError("FAISS index has not been created.")
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True,exist_ok=True)
        faiss.write_index(self.index,str(output_file))

    def load_index(self,file_path: str) -> None:
        input_file = Path(file_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Index file not found: {input_file}")

        self.index = faiss.read_index(str(input_file))

        self.embedding_dimension = self.index.d

    def search(self,query_embedding: np.ndarray,top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("FAISS index has not been loaded.")
        query_embedding = np.array([query_embedding],dtype=np.float32)
        distances, indices = self.index.search(query_embedding,top_k)
        return distances[0], indices[0]