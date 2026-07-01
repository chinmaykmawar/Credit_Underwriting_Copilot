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
    
    def get_structured_document(self, pdf_path: str, metadata:DocumentMetadata) -> Document:
        doc = fitz.open(pdf_path)
        structured_pages = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            # "blocks" returns a list of tuples: (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")

            page_text_pieces = []
            for b in blocks:
                block_text = b[4].strip()

            # Heuristic check: If lines have multiple numbers or look tabular
            # You can count spaces, tabs, or digits to flag potential tables
                lines = block_text.split('\n')
                digit_heavy_lines = sum(1 for line in lines if sum(c.isdigit() for c in line) > 5)
            
                if len(lines) > 2 and digit_heavy_lines / len(lines) > 0.4:
                # Wrap financial tables explicitly
                    page_text_pieces.append(f"\n[TABLE_START]\n{block_text}\n[TABLE_END]\n")
                else:
                    page_text_pieces.append(block_text)
                
            full_page_text = "\n\n".join(page_text_pieces)
            structured_pages[int(page_num + 1)]= str(full_page_text)
        return Document(metadata, structured_pages)