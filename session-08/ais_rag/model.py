import os
from typing import List, Dict, Any
from ollama import Client

class get_LLM:
    def __init__(self, host='http://localhost:11434', model='gemma:2b'):
        self.client = Client(host=host)
        self.model = model
        self.latest_response = None

    def generate(self, prompt: str, stream:str=False) -> str:
        try:
            response = self.client.generate(model=self.model, prompt=prompt, stream=stream)
            self.latest_response = response
            return response['response'] if 'response' in response else None
        except Exception as e:
            print(f"Error generating from Ollama: {e}")
            return None

    def __call__(self, prompt: str):
        return self.generate(prompt)
