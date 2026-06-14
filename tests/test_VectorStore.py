from src.retrieval.VectorStore import VectorStore
from tests.helper import get_saved_chunks

chunks=get_saved_chunks()

vector_store = VectorStore()
vector_store.create_index(chunks)
vector_store.save_index(r"data\processed\faiss.index")
vector_store.load_index(r"data\processed\faiss.index")

query_embedding = chunks[100].embedding

assert query_embedding is not None

distances, indices = vector_store.search(query_embedding,top_k=5)
print(indices)
print(distances)

for idx in indices:
    print("=" * 100)
    print(chunks[idx].start_page,chunks[idx].end_page)
    print(chunks[idx].text[:300])