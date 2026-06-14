from src.retrieval.embedder import Embedder
from src.chunking.models import Chunk
from tests.helper import get_chunks

embedder = Embedder()

chunks : list[Chunk]=get_chunks()
embedder.generate_embeddings(chunks)

embedder.save_chunks(chunks,r"data\processed\chunks.pkl")

loaded_chunks= embedder.load_chunks(r"data\processed\chunks.pkl")


assert chunks[0].embedding is not None

print(type(chunks[0].embedding))
print(chunks[0].embedding.shape)
print(chunks[0].embedding[:10])


assert loaded_chunks[0].embedding is not None

print(len(loaded_chunks))
print(loaded_chunks[0].embedding.shape)