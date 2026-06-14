from pathlib import Path
import re
import fitz
from .models import Document, DocumentMetadata, Page

class PDFReader:
    
    @staticmethod
    def _clean_text(text: str):
        text = text.strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def read_pdf(self,pdf_path: str,metadata: DocumentMetadata,clean_text: bool = True):
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")

        document = Document(metadata=metadata)

        with fitz.open(pdf_file) as pdf:
            for page_number in range(1, len(pdf)+1):
                page = pdf[page_number-1]
                text  = str(page.get_text("text"))
                if clean_text:
                    text = self._clean_text(text)
                document.pages[page_number] = text
        return document