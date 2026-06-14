from src.ingestion.models import DocumentMetadata, Document
from src.ingestion.pdf_reader import PDFReader
from src.chunking.chunker import Chunker
from src.chunking.models import Chunk
from src.retrieval.embedder import Embedder
from src.retrieval.VectorStore import VectorStore

def get_document() -> Document:
    metadata = DocumentMetadata(
        company_name="Tata Motors",
        document_type="Annual Report",
        source_category="External",
        source_file="Annual_Report.pdf",
        financial_year=2025
    )

    reader = PDFReader()

    document = reader.read_pdf(
        pdf_path=r"C:\Data\Python\AI Credit Underwriting Project\data\raw\sample_documents\tata_motors\tata-motor-IAR-2024-25.pdf",
        metadata=metadata
    )

    return document


def get_chunks()-> list[Chunk]:
    document=get_document()
    chunker = Chunker()
    chunks = chunker.create_chunks(document)
    return chunks

embedder = Embedder()

def get_embedder():
    return embedder

def get_chunks_with_embeddings():
    chunks : list[Chunk]=get_chunks()
    embedder.generate_embeddings(chunks)

def save_chunks(chunks, fileLoc =r"data\processed\chunks.pkl"):
    embedder.save_chunks(chunks,fileLoc)

def get_saved_chunks(fileLoc =r"data\processed\chunks.pkl"):
    return embedder.load_chunks(fileLoc)

vector_store=VectorStore()
vector_store.load_index(r"data\processed\faiss.index")

def get_VectorStore():
    return vector_store