from src.retrieval.retriever import Retriever, RetrievalResult
from src.llm.qwen_client import QwenClient

class RAGService:
    def __init__(self,retriever: Retriever,llm_client: QwenClient):
        self.retriever = retriever
        self.llm_client = llm_client

    def ask(self,question: str,top_k: int = 5) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        #retrieved_chunks = self.retriever.retrieve(query=question,top_k=top_k)
        results = self.retriever.retrieve_with_dis(query=question,top_k=top_k)

        print("\nRETRIEVAL RESULTS\n")

        for res in results:

            print("=" * 80)
            print(res.distance)
            print(res.chunk.start_page, res.chunk.end_page)
            print(res.chunk.text[:200])

        # for chunk in retrieved_chunks:
        #     print("=" * 100)
        #     print(chunk.start_page, chunk.end_page)
        #     print(chunk.text[:300])
        retrieved_chunks=[res.chunk for res in results]
        context = self._build_context(retrieved_chunks)
        prompt = self._build_prompt(question,context)
        print(prompt)
        return self.llm_client.generate_response(prompt)

    def _build_context(self,chunks) -> str:
        context_parts = []
        for chunk in sorted(chunks,key=lambda x: x.start_page):
            context_parts.append(chunk.text)
        return "\n\n".join(context_parts)

    def _build_prompt(self,question: str,context: str) -> str:
        return f"""
You are an experienced corporate banking and credit risk analyst.

Answer the user's question using only the information provided in the context.

If the answer is not available in the context, clearly state:
"The information is not available in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""