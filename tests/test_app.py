import sys
from pathlib import Path
sys.path.append(r'C:\Data\Python\AI Credit Underwriting Project')

import logging
logging.basicConfig(
    level=logging.INFO,
    format=" %(relativeCreated)d [%(levelname)s] %(filename)s -> %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"  # Clean, readable date/time format
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

logger.info("logger created")

from src.CreditUnderwritingApp import CreditUnderwritingApp
logger.info("CUApp imported")

app=CreditUnderwritingApp()
logger.info("CUApp loaded")

def test_doc_ingestion():
    logger.info("test started")
    doc1=r'C:\Data\Python\AI Credit Underwriting Project\data\raw\sample_documents\tata_motors'
    res=app.ingest_documents([doc1])
    if res:
        print('Updated documents')

def test_rag_response():
    logger.info("execution started")
    res=app.ask("What are the principal risks faced by Tata Motors?",top_k=10)
    print (res.answer)
    print (f'\n\n\n{res.sources}')
    

#test_rag_response()
test_doc_ingestion()