from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.chunking.models import Chunk

import logging
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self,model_name : str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self,chunks: list[Chunk],batch_size: int = 32) -> list[Chunk]:
        logger.info(f'Embedder.Generate Emb<>Entering')
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts,batch_size=batch_size,show_progress_bar=True,convert_to_numpy=True)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        logger.info(f'Embedder.Generate Emb<>Exiting')
        return chunks
    
    def generate_query_embedding(self, query : str):
        return self.model.encode(query,convert_to_numpy=True)