import os
from typing import List, Dict, Any
from ollama import chat, ChatResponse

class get_LLM:
    def __init__(self, model='gemma3'):
        self.model = model
        self.latest_response = None

    def generate(self, prompt: str) -> str:
        try:
            # Send the prompt to the Ollama model and receive a response
            response: ChatResponse = chat(model=self.model, 
                                          messages=[{'role': 'user', 'content': prompt}])

            # Extract the content of the assistant's reply
            self.latest_response = response.message.content
            return self.latest_response
        except Exception as e:
            print(f"Error generating from Ollama: {e}")
            return None

    def __call__(self, prompt: str):
        return self.generate(prompt)


if __name__ == "__main__":
    # Example usage
    os.system('cls' if os.name == 'nt' else 'clear')
    llm = get_LLM('gemma3')
    prompt = "What is the capital of France?"
    response = llm.generate(prompt)
    print(response)