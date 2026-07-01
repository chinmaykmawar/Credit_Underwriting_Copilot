from src.chunking.chunker import Chunker
from src.chunking.models import Chunk
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

document=get_document()
chunker = Chunker()
chunks = chunker.create_chunk_files(document)

print(f"Total Chunks: {len(chunks)}")

print("=" * 100)

empty_chunks = [chunk for chunk in chunks if len(chunk.text.strip()) == 0]

print(f'no of empty chunks :{len(empty_chunks)}')

small_nonempty_chunks = [chunk for chunk in chunks if (len(chunk.text) < 100 and len(chunk.text)>0)]

print ('\n\n\n\nSmall Non-empty chunks, 0<len<100 ')
for i in range(10):
    print("=" * 80)
    print(small_nonempty_chunks[i].start_page, small_nonempty_chunks[i].end_page)
    print(repr(small_nonempty_chunks[i].text))

small_chunks = [chunk for chunk in chunks if len(chunk.text) < 100]


print ('\n\n\n\nSmall chunks, len<100 ')
for i in range(10):
    print("=" * 80)
    print(small_chunks[i].start_page, small_chunks[i].end_page)
    print(repr(small_chunks[i].text))







# lengths = [len(chunk.text) for chunk in chunks]

# print(f'shortest chunk : {min(lengths)}')
# print(f'longest chunk : {max(lengths)}')
# print(f'average chunk : {sum(lengths) / len(lengths)}')



# print(f'no of small chunks :{len(small_chunks)}')