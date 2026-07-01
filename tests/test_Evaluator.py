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

from src.rag.rag_service import RAGService
from src.retrieval.VectorStore import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.embedder import Embedder
from src.chunking.chunker import Chunker
from src.llm.qwen_client import QwenClient
from src.evaluation.rag_evaluator import LLMJudgeEvaluator
from src.CreditUnderwritingApp import CreditUnderwritingApp

embedder=Embedder()
llm_client=QwenClient()
evaluator = LLMJudgeEvaluator(llm_client)
chunker=Chunker()
vs=VectorStore()
retriever=Retriever(vs,chunker)
qwenClient=QwenClient()
rag_service=RAGService(embedder, retriever, qwenClient)

questions=["Did Tata Motors report a profit of $100 Billion in 2025?",
            "What is the specific recipe for the cafeteria food at Tata Motors' headquarters",
            "What are the financial risks faced by Tata Motors?",
            "What was the profit made by Tata Motors?",
            "Which of Tata Motors investments did not produce desired returns?"]

#questions=[ "What are the financial risks faced by Tata Motors?"]

output=[]
reasons=[]
output_file=Path(r'Evaluator_test.output')

def ask(question, top_k=10):
    query=question.lower().strip()
    query_embeddings=embedder.generate_query_embedding(query)
    
    results=retriever.retrieve("Tata Motors", 2025, query_embeddings,top_k=top_k)
    if len(results)>0:
        retrieved_chunks=[res.chunk for res in results]
        prompt = rag_service._get_system_prompt() +"\n\n"+ rag_service._get_user_prompt(question,retrieved_chunks)
        answer= qwenClient.generate_response(prompt)
        sources = [c.text for c in  retrieved_chunks]
        with open(output_file, 'a', encoding='utf-8') as f:
            #f.write(f'\n prompt:{prompt}')
            f.write(f'\n answer:{answer}')
            for i,result in enumerate(results):
                f.write(f'\n Chunk {i} -> {result.chunk.metadata.source_file}: Page {result.chunk.start_page} to Page {result.chunk.end_page}')
                f.write(f'\n {result.chunk.text}')
        return (answer, sources)
    else:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('No Chunks')
        return ('No details found for mentioned companies for years mention in the query',[])

def run_evaluator(question):
    (answer, sources)=ask(question=question, top_k=10)

    if sources:
        evaluation=evaluator.run_full_triad(question,answer,sources)
        for param, result in evaluation.items():
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(param+"->"+str(result['score'])+'\n')
            if result['score']!=1:
                reasons.append (f'{question}//{str(param)}//{str(result['score'])}/n/n{result['reasoning']}')

with open(output_file,"w", encoding='utf-8') as f:
    f.write("execution started")

for question in questions:      
    with open(output_file,'a', encoding='utf-8') as f:
        f.write(f"\n\n Question = {question}")
    run_evaluator(question)

with open(output_file,'a', encoding='utf-8') as f:
        f.write(f'\n{['=']*100}\n')
        f.writelines(reasons)



