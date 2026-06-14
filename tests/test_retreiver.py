from tests.helper import get_saved_chunks, get_embedder, get_VectorStore
from src.retrieval.retriever import Retriever, RetrievalResult

retriever = Retriever(
    embedder=get_embedder(),
    vector_store=get_VectorStore(),
    chunks=get_saved_chunks()
)

texts=["principal risks faced by Tata Motors","market risks commodity risks supply chain risks", "Principal risks"]
    
for text in texts:
    results : list[RetrievalResult]  = retriever.retrieve(text,top_k=5)       

    print (f'\n\n\n\n{text}')

    for res in results:
        print("=" * 100)
        print(f'Distance : {res.distance}')
        print(f"Pages: {res.chunk.start_page}-{res.chunk.end_page}")
        print(res.chunk.text[:500]) 