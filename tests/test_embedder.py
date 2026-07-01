from src.retrieval.embedder import Embedder
from src.chunking.models import Chunk
from src.chunking.chunker import Chunker
from src.ingestion.models import Document, DocumentMetadata
from src.ingestion.pdf_reader import PDFReader



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

def get_chunks()-> list[Chunk]:
    document=get_document()
    chunker = Chunker()
    chunks = chunker.create_chunk_files(document)
    return chunks

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