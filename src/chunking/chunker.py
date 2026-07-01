from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from src.ingestion.models import Document, DocumentMetadata
from src.chunking.models import Chunk
from src.retrieval.embedder import Embedder
from pathlib import Path
import pickle
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)
logger.info(f'Chunker.create_chunk_files<>Entering for doc')

class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, base_folder=r"data\processed"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_index_folder=Path(base_folder)
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def create_chunk_files(self,document: Document, embedder:Embedder) -> None:
        logger.info(f'Chunker.create_chunk_files<>Entering for doc:{document.metadata.company_name}')
        if not self._check_existing_data(document.metadata.source_file, document.metadata.company_name, document.metadata.financial_year):
            print ('Doc already exists')
            return
        chunks=self._get_chunks_from_Document(document)
        if not chunks:
            return
        chunks=embedder.generate_embeddings(chunks)
        self._save_chunks(chunks, document.metadata.company_name, document.metadata.financial_year)
        logger.info(f'Chunker.create_chunk_files<>Exiting for doc:{document.metadata.company_name}')

    def load_chunks(self,company_name:str, financial_year:int) -> list[Chunk]:
        logger.info(f'Chunker.load Chunks<>Entering for doc:{company_name}')
        input_file = self._get_isolated_paths(company_name, financial_year)
        if not input_file.exists():
            return []
        with open(input_file, "rb") as f:
            chunks = pickle.load(f)
        logger.info(f'Chunker.load Chunks<>Exiting No of chunks:{len(chunks)}')
        return chunks
    
    def _get_chunks_from_Document(self, document:Document):
        pages=document.pages
        chunks=[]
        lastChunk=Chunk('',0,0,document.metadata)
        for i in range(1,len(pages)+1):
            currPageText=lastChunk.text+'\n\n'+pages[i]
            currChunks=self._split_text(currPageText, i, document.metadata)
            if not currChunks:
                continue
            currChunks[0].start_page=lastChunk.start_page
            if (len(lastChunk.text)>self.chunk_size-self.chunk_overlap) and len(currChunks)>1:
                currChunks[0].end_page=lastChunk.start_page
                currChunks[1].start_page=lastChunk.start_page
            chunks.extend(currChunks)
            lastChunk:Chunk =chunks.pop()
        chunks.append(lastChunk)
        return chunks

    def _check_existing_data(self, doc_name, c_name:str, f_year:int)->bool:
        existing_chunks=self.load_chunks(c_name, f_year)
        existing_docs=[]
        for chunk in existing_chunks:
            if chunk.metadata.source_file not in existing_docs:
                existing_docs.append(chunk.metadata.source_file)
        if doc_name in existing_docs:
            return False
        else:
            return True

    def _split_text(self,text:str,page_num:int, metadata:DocumentMetadata)->list[Chunk]:
        chunks=[]
        if text.startswith("[TABLE_CONTD.]"):
            end_tag='[TABLE_END]'
            index=text.find(end_tag)
            if index==-1:
                end_tag='[TABLE_CONTD.]'
                index=text.find(end_tag)
            chunks.append(Chunk(text=text[:index+len(end_tag)],start_page=page_num,end_page=page_num, metadata=metadata))
            text=text[index+len(end_tag):]
            
        pattern = re.compile(r"(\[TABLE_START\].*?\[TABLE_END\])", re.DOTALL)
        parts = pattern.split(text)
        
        for part in parts:
            part=part.strip()
            if not part:
                continue
            if part.startswith("[TABLE_START]"):
                # If the table fits within our chunk envelope, keep it whole
                if len(part) <= self.chunk_size:
                    chunks.append(Chunk(part,start_page=page_num, end_page=page_num,metadata=metadata))
                else:
                    clean_table = part.replace("[TABLE_START]", "").replace("[TABLE_END]", "").strip()
                    sub_parts = self.splitter.split_text(clean_table)
                    for i,sub in enumerate(sub_parts):
                        if i==0:
                            chunks.append(Chunk(text=f"[TABLE_START]\n{sub}\n[TABLE_CONTD.]",start_page=page_num, end_page=page_num,metadata=metadata))    
                        elif i==len(sub_parts)-1:
                            chunks.append(Chunk(text=f"[TABLE_CONTD.]\n{sub}\n[TABLE_END]",start_page=page_num, end_page=page_num,metadata=metadata))
                        else:
                            chunks.append(Chunk(text=f"[TABLE_CONTD.]\n{sub}\n[TABLE_CONTD.]",start_page=page_num, end_page=page_num,metadata=metadata))
            else:
                chunks.extend([Chunk(text=p,start_page=page_num, end_page=page_num,metadata=metadata) for p in self.splitter.split_text(part)])
        return chunks
    
    def _save_chunks(self,chunks: list[Chunk],company_name: str, fiscal_year: int) -> None:
        output_file = self._get_isolated_paths(company_name, fiscal_year)
        if output_file.exists():
            with open(output_file, "rb") as f:
                existing_chunks = pickle.load(f)
                chunks = existing_chunks + chunks 

        with open(output_file, "wb") as f:
            pickle.dump(chunks, f)

    def _get_isolated_paths(self, company_name: str, fiscal_year: int) -> Path:
        sanitized_company = company_name.strip().lower().replace(" ", "_")
        folder_path = self.base_index_folder.joinpath(sanitized_company,str(fiscal_year))
        folder_path.mkdir(parents=True,exist_ok=True)
        index_path = folder_path.joinpath("chunks.pkl")
        return index_path