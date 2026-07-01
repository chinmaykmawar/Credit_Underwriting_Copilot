from src.chunking.models import Chunk
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever, RetrievalResult
from src.rag.models import RAGResponse,SourceDetails
from src.llm.qwen_client import QwenClient

class RAGService:
    def __init__(self,embedder: Embedder, retriever: Retriever,llm_client: QwenClient):
        self.embedder = embedder
        self.retriever = retriever
        self.llm_client = llm_client

    def ask(self,company_names:list[str], financial_years:list[int], question: str,top_k: int = 10) -> RAGResponse:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        query=question.lower().strip()
        query_embeddings=self.embedder.generate_query_embedding(query)
        results=[]
        for company_name in company_names:
            for financial_year in financial_years:
                results.extend(self.retriever.retrieve(company_name, financial_year, query_embeddings,top_k=top_k))
                results = sorted(results, key=lambda x: x.distance, reverse=True)[:top_k]
        if len(results)>0:
            retrieved_chunks=[res.chunk for res in results]
            prompt = self._get_system_prompt() +"\n\n"+ self._get_user_prompt(question,retrieved_chunks)
            answer= self.llm_client.generate_response(prompt)
            sources = self._extract_sources(retrieved_chunks)
            return RAGResponse(answer, sources)
        else:
            return RAGResponse('No details found for mentioned companies for years mention in the query', [])
            

    def _build_context(self,chunks) -> str:
        context_parts = []
        for chunk in sorted(chunks,key=lambda x: x.start_page):
            context_parts.append(
f"""
[Source: {chunk.metadata.document_type}
Pages: {chunk.start_page}-{chunk.end_page}]

{chunk.text}
""".strip())
        return "\n\n".join(context_parts)

    def _extract_sources(self, chunks) -> list[SourceDetails]:
        source_files = {}

        for chunk in chunks:
            source = SourceDetails(
                company_name=chunk.metadata.company_name,
                document_type=chunk.metadata.document_type,
                source_file=chunk.metadata.source_file,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
            )
            source_files.setdefault(source.source_file, []).append(source)
        unique_sources = []

        for sources in source_files.values():
            sources.sort(key=lambda s: s.start_page)
            merged = sources[0]

            for source in sources[1:]:
                # Merge only if ranges overlap or touch
                assert merged is not None
                if merged.end_page >= source.start_page:
                    merged = SourceDetails.combineSources(merged, source)
                else:
                    unique_sources.append(merged)
                    merged = source
            unique_sources.append(merged)
        return unique_sources

    def _get_user_prompt(self,question: str,retrieved_chunks: list[Chunk]) -> str:
        return f"""

------------------------
CONTEXT
------------------------

{self._build_context(retrieved_chunks)}

------------------------
QUESTION
------------------------

{question}

------------------------
ANSWER
------------------------
""".strip()
    
    def _get_system_prompt(self) -> str:
        return """
You are a senior Corporate Banking and Credit Risk Analyst.

You analyze annual reports, investor presentations, credit rating reports,
financial statements and other corporate documents.

Use ONLY the information provided in the context.

If multiple context sections discuss the same topic, combine them into a single coherent answer.

Do not omit relevant information simply because it appears in different context sections.

If the context is incomplete, explicitly mention any uncertainty rather than making assumptions.

Never use outside knowledge.

If the answer cannot be found in the provided context, reply exactly:

"The information is not available in the provided documents."

Write concise, professional responses suitable for a corporate credit memorandum.

Format every response using Markdown.

Use the following structure:

## Executive Summary
Provide a brief answer in 2-5 sentences.

## Key Findings
Present the important findings as bullet points.
If the information is naturally tabular, use a Markdown table instead.

## Supporting Evidence
Summarize the evidence from the supplied context.
Do not invent facts or cite information outside the context.
""".strip()