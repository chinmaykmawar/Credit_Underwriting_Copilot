from src.ingestion.models import DocumentMetadata, Document
from src.ingestion.pdf_reader import PDFReader

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

print(document.pages[50][:500])
print("=" * 100)

print(document.pages[200][:500])
print("=" * 100)

print(document.pages[400][:500])