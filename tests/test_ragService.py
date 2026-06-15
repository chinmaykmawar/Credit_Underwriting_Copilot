from src.ingestion.pdf_reader import PDFReader
from src.ingestion.models import Document, DocumentMetadata
from src.chunking.models import Chunk
from src.chunking.chunker import Chunker
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever, RetrievalResult
from src.retrieval.VectorStore import VectorStore
from src.llm.qwen_client import QwenClient
from src.rag.rag_service import RAGService
from pathlib import Path

embedder=Embedder()
def get_document() -> Document:
    metadata = DocumentMetadata(
        company_name="Tata Motors",
        document_type="Annual Report",
        source_category="External",
        source_file="Annual_Report.pdf",
        financial_year=2025
    )
    reader = PDFReader()
    document = reader.read_pdf(pdf_path=r"C:\Data\Python\AI Credit Underwriting Project\data\raw\sample_documents\tata_motors\tata-motor-IAR-2024-25.pdf",metadata=metadata)
    return document

def get_chunks():
    if Path(r"data\processed\chunks.pkl").exists():
        chunks=embedder.load_chunks()
    else:
        document=get_document()
        chunker = Chunker()
        chunks = chunker.create_chunks(document)
        embedder.generate_embeddings(chunks)
        embedder.save_chunks(chunks)
    return chunks

chunks=get_chunks()

def get_VectorStore():
    vector_store=VectorStore()
    if not Path(r"data\processed\faiss.index").exists():
        vector_store.create_index(chunks)
        vector_store.save_index()
    vector_store.load_index(r"data\processed\faiss.index")
    return vector_store

vector_store=get_VectorStore()

retriever = Retriever(embedder=embedder,vector_store=vector_store,chunks=chunks)

qwen=QwenClient()
rag_service = RAGService(retriever=retriever,llm_client=qwen)
response = rag_service.ask(
    "What are the principal risks faced by Tata Motors?",
    top_k=10
)
print(response)