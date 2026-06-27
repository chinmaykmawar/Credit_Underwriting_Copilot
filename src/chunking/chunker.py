from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from src.ingestion.models import Document
from src.chunking.models import Chunk
from pathlib import Path
import pickle

class Chunker:

    def create_chunks(self,document: Document,chunk_size: int = 1000,chunk_overlap: int = 100) -> list[Chunk]:
        combined_text = self._build_document_text(document)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        text_chunks = splitter.split_text(combined_text)
        chunks = []
        search_start = 0
        for text in text_chunks:
            start_idx = combined_text.find(text, search_start)
            if start_idx == -1:
                raise ValueError(f"Could not find text:{text} in combined text")
            end_idx = start_idx + len(text)

            start_page, end_page = self._get_page_numbers(combined_text,start_idx, end_idx)
            cleaned_chunk_text = self._remove_page_markers(text)
            cleaned_chunk_text = cleaned_chunk_text.strip()

            if not cleaned_chunk_text:
                continue

            if len(cleaned_chunk_text) < 20:
                continue

            search_start = start_idx + 1
            chunks.append(Chunk(text=cleaned_chunk_text,start_page=start_page,end_page=end_page,metadata=document.metadata))
            print(f'Chunk with start Page:{start_page} appended')
        return chunks

    def _build_document_text(self,document: Document) -> str:
        text = []
        for page_number in sorted(document.pages.keys()):
            text.append(f"\n\n-!PAGE {page_number}!-\n\n")
            text.append(document.pages[page_number])
        return "".join(text)

    def _get_page_numbers(self,text: str,start_idx: int, end_idx:int) -> tuple[int,int]:
        search_start = start_idx+5
        pageNoText_start = text.find("-!PAGE ",search_start)
        pageNoText_end = text.find("!-",pageNoText_start)
        pageNoText = text[pageNoText_start + 7:pageNoText_end]
        start_page = int(pageNoText) -1
        end_page=start_page+1
        while True:
            pageNoText_start = text.find("-!PAGE ",pageNoText_end+1)
            if pageNoText_start>end_idx or pageNoText_start==-1:
                break
            pageNoText_end = text.find("!-",pageNoText_start+1)
            if pageNoText_end>end_idx:
                break
            pageNoText = text[pageNoText_start + 7:pageNoText_end]
            end_page = int(pageNoText)
        return (start_page,end_page)

    def _remove_page_markers(self,text: str) -> str:
        return re.sub(r"-!PAGE\s+\d+!-","",text).strip()
    
    def save_chunks(self,chunks: list[Chunk],file_path: str =r"data\processed\chunks.pkl") -> None:
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True,exist_ok=True)

        with open(output_file, "wb") as f:
            pickle.dump(chunks, f)

    def load_chunks(self,file_path: str=r"data\processed\chunks.pkl") -> list[Chunk]:
        input_file = Path(file_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {input_file}")
        with open(input_file, "rb") as f:
            chunks = pickle.load(f)
        return chunks