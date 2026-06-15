from src.ingestion.pdf_reader import PDFReader
from src.ingestion.models import Document, DocumentMetadata
from src.chunking.models import Chunk
from src.chunking.chunker import Chunker
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever, RetrievalResult
from src.retrieval.VectorStore import VectorStore
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

texts=["What are the principal risks faced by Tata Motors?", "Principal risks", "List Tata Motors principal risks"]
    
for text in texts:
    results : list[RetrievalResult]  = retriever.retrieve_with_dis(text,top_k=5)       

    print (f'\n\n\n\nInput Text:{text}')

    for res in results:
        print("=" * 100)
        print(f'Distance : {res.distance}')
        print(f"Pages: {res.chunk.start_page}-{res.chunk.end_page}")
        print(res.chunk.text[:500]) 

# chunks= get_saved_chunks()
# for chunk in chunks:
#     if chunk.start_page == 73:
#         print(chunk.text)
        
#     if "trade tensions" in chunk.text.lower():
#         print(chunk.start_page, chunk.end_page)
#         print(chunk.text[:1000])
        
#     if "global challenges" in chunk.text.lower():
#         print(chunk.start_page, chunk.end_page)
        
#     if "principal risks" in chunk.text.lower():
#         print(chunk.start_page, chunk.end_page)
#         print(chunk.text[:1000])