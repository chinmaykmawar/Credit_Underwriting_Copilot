from src.chunking.chunker import Chunker
from src.retrieval.VectorStore import VectorStore

vs= VectorStore()
chunker=Chunker()
(company_name, financial_year)=('Tata Motors', 2025)

index = vs._load_index(company_name, financial_year)
chunks = chunker.load_chunks(company_name, financial_year)

print(f"FAISS Total Vectors: {index.ntotal}")
print(f"Pickle Total Chunks: {len(chunks)}")