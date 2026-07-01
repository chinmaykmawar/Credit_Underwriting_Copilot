from pathlib import Path
from src.ingestion.pdf_reader import PDFReader
from src.ingestion.models import Document, DocumentMetadata
from src.chunking.models import Chunk
from src.chunking.chunker import Chunker
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever, RetrievalResult
from src.retrieval.VectorStore import VectorStore
from src.llm.qwen_client import QwenClient
from src.rag.rag_service import RAGService,RAGResponse
import logging
logger =logging.getLogger(__name__)

class CreditUnderwritingApp:
    def __init__(self):
        self.embedder= Embedder()
        logger.info('Embedder Created')        
        self.chunker=Chunker()
        logger.info('Chunker Created')        
        self.qwen=QwenClient()
        logger.info('Qwen Created')        
        self.vectorStore=VectorStore()
        logger.info('VectorStore Created')        
        self.retriever = Retriever(self.vectorStore, self.chunker)
        logger.info('Retreiver Created')        
        self.rag_service = RAGService(self.embedder, self.retriever, self.qwen)
        logger.info('RAG Service Created')        

    def ask(self, question: str, top_k: int = 10)->RAGResponse:
        if self.rag_service is None:
            raise RuntimeError("Knowledge base not initialized. Call ingest_documents() first.")
        (company_names, financial_years)=(['Tata Motors'], [2025])
        return self.rag_service.ask(company_names, financial_years,question, top_k)

    def ingest_documents(self, document_folders: list[str])-> bool:
        documents=self._load_documents(document_folders)
        chunks=[]
        for doc in documents:
            self.chunker.create_chunk_files(doc,self.embedder)
            chunks.extend(self._load_chunks(doc.metadata.company_name, doc.metadata.financial_year))
        return self._build_vector_store(chunks)
        #self._initialize_objects()
    
    def _load_documents(self, documentFolders:list[str])->list[Document]:
        reader = PDFReader()
        documents=[]
        for documentFolder in documentFolders:
            folder = Path(documentFolder)
            for pdf_file in folder.glob("*.pdf"):
                document = reader.get_structured_document(pdf_path=str(pdf_file),metadata=self._get_Metadata(pdf_file))
                documents.append(document)
        return documents
        
    def _get_Metadata(self, pdf_file:Path)->DocumentMetadata:
        company_name=pdf_file.parent.name
        source_file=pdf_file.name
        document_type, year = pdf_file.stem.rsplit("-", maxsplit=1)
        return DocumentMetadata(company_name, document_type, 'External', source_file,int(year))

    def _load_chunks(self, company_name:str, financial_year:int)->list[Chunk]:
            chunks=self.chunker.load_chunks(company_name, financial_year)
            print("Chunks loaded from Memory")
            return chunks
    
    def _create_chunks(self, documentFolders:list[str]):
        print ("Creating Chunks.....")
        documents=self._load_documents(documentFolders)
        chunks=[]
        for doc in documents:
            currChunks=self.chunker.create_chunk_files(doc, self.embedder)
            if currChunks:
                chunks.extend(currChunks)
                self.embedder.generate_embeddings(currChunks)
                self.chunker._save_chunks(currChunks, doc.metadata.company_name, doc.metadata.financial_year)
        print("Chunks created and saved")        

    def _build_vector_store(self, chunks)-> bool:
        vector_store=VectorStore()
        print ("Creating Index file...")
        return vector_store.create_index_file(chunks)