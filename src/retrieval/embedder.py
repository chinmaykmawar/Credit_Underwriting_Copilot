from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from src.chunking.models import Chunk

class Embedder:
    def __init__(self,model_name : str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self,chunks: list[Chunk],batch_size: int = 32) -> None:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts,batch_size=batch_size,show_progress_bar=True,convert_to_numpy=True)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

    def save_chunks(self,chunks: list[Chunk],file_path: str =r"data\processed\chunks.pkl") -> None:
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True,exist_ok=True)

        with open(output_file, "wb") as f:
            pickle.dump(chunks, f)

    def load_chunks(self,file_path: str=r"data\processed\chunks.pkl") -> list[Chunk]:
        input_file = Path(file_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {input_file}")
        with open(input_file, "rb") as f:
            chunks = pickle.load(f)
        return chunks
    
    def generate_query_embedding(self, query : str):
        return self.model.encode(query,convert_to_numpy=True)