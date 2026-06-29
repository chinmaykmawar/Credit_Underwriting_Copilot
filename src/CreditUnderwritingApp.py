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

class CreditUnderwritingApp:
    def __init__(self):
        self.embedder= Embedder()
        self.chunker=Chunker()
        self.qwen=QwenClient()
        self.vectorStore=None
        self.retriever = None
        self.rag_service = None

        if Path(r"data\processed\chunks.pkl").exists() and Path(r"data\processed\faiss.index").exists():
            chunks=self._load_chunks()
            self._initilize_objects(chunks)
        
        elif Path(r"data\processed\chunks.pkl").exists():
            chunks=self._load_chunks()
            self._build_vector_store(chunks)
            self._initilize_objects(chunks)

    def ask(self, question: str, top_k: int = 10)->RAGResponse:
        if self.rag_service is None:
            raise RuntimeError("Knowledge base not initialized. Call ingest_documents() first.")
        return self.rag_service.ask(question, top_k)

    def ingest_documents(self, document_folders: list[str]):
        docuemnts=self._load_documents(document_folders)
        chunks=[]
        for doc in docuemnts:
            chunks.extend(self.chunker.create_chunks(doc))
        self.embedder.generate_embeddings(chunks)
        self.chunker.save_chunks(chunks)
        chunks=self._load_chunks()
        self._build_vector_store(chunks)
        self._initilize_objects(chunks)
        
    def _initilize_objects(self, chunks):
        self.vectorStore=self._load_vector_store()
        self.retriever = Retriever(vector_store=self.vectorStore,chunks=chunks)
        self.rag_service = RAGService(embedder=self.embedder, retriever=self.retriever,llm_client=self.qwen) 

    def _get_Metadata(self, pdf_file:Path)->DocumentMetadata:
        company_name=pdf_file.parent.name
        source_file=pdf_file.name
        document_type, year = pdf_file.stem.rsplit("-", maxsplit=1)
        return DocumentMetadata(company_name, document_type, 'External', source_file,int(year))

    def _load_documents(self, documentFolders:list[str])->list[Document]:
        reader = PDFReader()
        documents=[]
        for documentFolder in documentFolders:
            folder = Path(documentFolder)
            for pdf_file in folder.glob("*.pdf"):
                document = reader.read_pdf(pdf_path=str(pdf_file),metadata=self._get_Metadata(pdf_file))
                documents.append(document)

        return documents
        
    def _load_chunks(self)->list[Chunk]:
            chunks=self.chunker.load_chunks()
            print("Chunks loaded from Memory")
            return chunks
    
    def _create_chunks(self, documentFolders:list[str]):
        print ("Creating Chunks.....")
        documents=self._load_documents(documentFolders)
        chunks=[]
        for doc in documents:
            chunks.extend(self.chunker.create_chunks(doc))
        self.embedder.generate_embeddings(chunks)
        self.chunker.save_chunks(chunks)
        print("Chunks created and saved")        

    def _build_vector_store(self, chunks):
        vector_store=VectorStore()
        print ("Creating Index file...")
        vector_store.create_index(chunks)
        vector_store.save_index()

    def _load_vector_store(self)->VectorStore:
        vector_store=VectorStore()
        vector_store.load_index(r"data\processed\faiss.index")
        print ("Indexes loaded")
        return vector_store
