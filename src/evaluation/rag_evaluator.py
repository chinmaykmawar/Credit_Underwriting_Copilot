import json
import logging
from typing import List
import re

# Set up basic logging for the evaluation runs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMJudgeEvaluator:
    def __init__(self, llm_client):
        """
        Takes an initialized LLM client (e.g., your local Ollama/Qwen wrapper).
        The client should have a method like `.chat(system_prompt, user_prompt)`
        """
        self.llm = llm_client

    def _parse_json(self, response_text: str) -> dict:
    # Use regex to find the first '{' and last '}'
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"score": 0, "reasoning": "Failed to parse JSON output from Judge LLM."}

    def evaluate_context_relevance(self, query: str, retrieved_contexts: List[str]) -> dict:
        system_prompt = """
        You are an expert evaluator grading a Retrieval-Augmented Generation (RAG) system.
        Your task is to evaluate CONTEXT RELEVANCE.
        Does the provided context contain the necessary information to answer the user's query?
        
        Output ONLY a valid JSON object with two keys:
        - "score": 1 if the context is relevant and sufficient, 0 if it is irrelevant or insufficient.
        - "reasoning": A 1-2 sentence explanation for your score.
        """
        
        context_str = "\n\n---\n\n".join(retrieved_contexts)
        user_prompt = f"Query: {query}\n\nRetrieved Context:\n{context_str}"
        
        response = self.llm.chat(system=system_prompt, user=user_prompt)
        return self._parse_json(response)

    def evaluate_groundedness(self, answer: str, retrieved_contexts: List[str]) -> dict:
        system_prompt = """
        You are an expert evaluator grading a Retrieval-Augmented Generation (RAG) system.
        Your task is to evaluate GROUNDEDNESS (Faithfulness).
        Is the generated answer fully supported by the provided context? If the answer contains ANY facts, numbers, or claims not present in the context, it is not grounded.
        
        Output ONLY a valid JSON object with two keys:
        - "score": 1 if fully grounded, 0 if it contains ungrounded claims or hallucinations.
        - "reasoning": A 1-2 sentence explanation for your score.
        """
        
        context_str = "\n\n---\n\n".join(retrieved_contexts)
        user_prompt = f"Context:\n{context_str}\n\nGenerated Answer:\n{answer}"
        
        response = self.llm.chat(system=system_prompt, user=user_prompt)
        return self._parse_json(response)

    def evaluate_answer_relevance(self, query: str, answer: str) -> dict:
        system_prompt = """
        You are an expert evaluator grading a Retrieval-Augmented Generation (RAG) system.
        Your task is to evaluate ANSWER RELEVANCE.
        Does the generated answer directly address and answer the user's query without unnecessary tangents?
        
        Output ONLY a valid JSON object with two keys:
        - "score": 1 if highly relevant, 0 if evasive, generic, or off-topic.
        - "reasoning": A 1-2 sentence explanation for your score.
        """
        
        user_prompt = f"Query: {query}\n\nGenerated Answer:\n{answer}"
        
        response = self.llm.chat(system=system_prompt, user=user_prompt)
        return self._parse_json(response)

    def run_full_triad(self, query: str, answer: str, contexts: list[str]) -> dict:
        """Executes all three evaluation metrics."""
        return {
            "context_relevance": self.evaluate_context_relevance(query, contexts),
            "groundedness": self.evaluate_groundedness(answer, contexts),
            "answer_relevance": self.evaluate_answer_relevance(query, answer)
        }