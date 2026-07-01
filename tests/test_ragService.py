import sys
from pathlib import Path


from src.ingestion.pdf_reader import PDFReader
from src.ingestion.models import Document, DocumentMetadata
from src.chunking.models import Chunk
from src.chunking.chunker import Chunker
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever, RetrievalResult
from src.retrieval.VectorStore import VectorStore
from src.llm.qwen_client import QwenClient
from src.rag.rag_service import RAGService

print ("execution started")

def get_document() -> Document:
    reader = PDFReader()
    metadata = DocumentMetadata(
        company_name="Tata Motors",
        document_type="Annual Report",
        source_category="External",
        source_file="Annual_Report.pdf",
        financial_year=2025
    )
    
    document = reader.read_pdf(pdf_path=r"C:\Data\Python\AI Credit Underwriting Project\data\raw\sample_documents\tata_motors\tata-motor-IAR-2024-25.pdf",metadata=metadata)
    return document

def get_chunks():
    chunker = Chunker()
    if Path(r"data\processed\chunks.pkl").exists():
        chunks=chunker.load_chunks()
        print("Chunks loaded from Memory")
    else:
        print ("Creating Chunks.....")
        document=get_document()
        chunks = chunker.create_chunk_files(document)
        embedder=Embedder()
        embedder.generate_embeddings(chunks)
        chunker._save_chunks(chunks)
        print("Chunks created and saved")
    return chunks

def get_VectorStore():
    vector_store=VectorStore()
    if not Path(r"data\processed\faiss.index").exists():
        print ("Creating Index file...")
        vector_store.create_index_file(chunks)
        vector_store.save_index()
    vector_store.load_index(r"data\processed\faiss.index")
    print ("Indexes loaded")
    return vector_store

chunks=get_chunks()
vectorStore=get_VectorStore()
retriever = Retriever(vector_store=vectorStore,chunks=chunks)
print("Retreiver object created")

qwen=QwenClient()
rag_service = RAGService(embedder=Embedder(), retriever=retriever,llm_client=qwen)
print("RAg_Service object created")
response = rag_service.ask("What are the principal risks faced by Tata Motors?",top_k=10)
with open("tests/output.txt", "w", encoding="utf-8") as f:
    f.write(str(response))
    f.write("\nSource\n")
    f.write("=" * 80)
    for source in response.sources:
        f.write(f"\n{source.document_type} | "+ f"Pages {source.start_page}-{source.end_page}")
