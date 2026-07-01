from ollama import chat

class QwenClient:
    def __init__(self,model_name: str = "qwen3:8b"):
        self.model_name = model_name

    def generate_response(self,prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        response = chat(model=self.model_name,messages=[{"role": "user","content": prompt}])
        return response["message"]["content"]
    
    def chat(self, system, user):
        response= chat(
            model=self.model_name,
            messages=[
                {"role": "system","content": system},
                {"role": "user","content": user}],
            format="json", 
            options={
                "temperature": 0.0 
                })
        return response["message"]["content"]