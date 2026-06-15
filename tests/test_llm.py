from src.llm.qwen_client import QwenClient

client = QwenClient()
response = client.generate_response("What is credit risk?")

print(response)