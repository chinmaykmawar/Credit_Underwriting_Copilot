from src.ingestion.models import DocumentMetadata, Document
from src.ingestion.pdf_reader import PDFReader
from src.chunking.chunker import Chunker
from src.chunking.models import Chunk
from src.retrieval.embedder import Embedder
from src.retrieval.VectorStore import VectorStore
from src.retrieval.retriever import Retriever, RetrievalResult
from src.llm.qwen_client import QwenClient





embedder = Embedder()

def get_embedder():
    return embedder

def regenrate_chunks():
    chunks : list[Chunk]=get_chunks()
    embedder.generate_embeddings(chunks)
    save_chunks(chunks)

def save_chunks(chunks, fileLoc =r"data\processed\chunks.pkl"):
    embedder.save_chunks(chunks,fileLoc)

def get_saved_chunks(fileLoc =r"data\processed\chunks.pkl"):
    return embedder.load_chunks(fileLoc)



chunks= get_saved_chunks()
assert chunks is not None
retreiver=Retriever(embedder, vector_store, chunks)
def get_retreiver():
    return retreiver
