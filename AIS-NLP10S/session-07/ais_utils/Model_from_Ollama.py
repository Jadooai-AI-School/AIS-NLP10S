import os
from ollama import Client

class get_chatLLM:
    def __init__(self, host='http://localhost:11434', model='gemma:2b'):
        self.client = Client(host=host)
        self.model = model
        self.latest_response = None

    def invoke(self, prompt):
        try:
            response = self.client.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            self.latest_response = response
            return response['message']['content'] if 'message' in response else None
        except Exception as e:
            print(f"Error invoking Ollama: {e}")
            return None

    def raw(self, prompt):
        try:
            response = self.client.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            self.latest_response = response
            return response
            
        except Exception as e:
            print(f"Error invoking Ollama: {e}")
            return None

    def __call__(self, prompt):
        # You can choose either invoke or raw here, depending on which behavior you want as the default
        return self.invoke(prompt) # Change this to self.invoke(prompt) if needed
##########################################################################################################################################
if __name__ == "__main__":
    llm = get_chatLLM()
    # clear screen for cleaner output
    os.system('cls' if os.name == 'nt' else 'clear') 
    
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    
    question_template = PromptTemplate(
        input_variables=["question"],
        template="{question}"
        )
    #prompt = question_template.format(question="Why is the sky blue?")
    prompt = question_template.invoke({"question":"Why is the sky blue?"})
        
    #response = llm.invoke(prompt)
    response = llm.invoke(prompt.to_string())
    
    print("[1] Invoke method response :", response) 
    print('\n\n')
    
    
    translation_template = PromptTemplate(
        input_variables=["text", "input_language", "output_language"],
        template="""
        Translate the following text from {input_language} to {output_language}.
        Text: {text}
        """)
    # Format the prompt
    prompt = translation_template.format(input_language="English", output_language="French", text="I love my cat")
    
    response = llm.invoke(prompt)
    print("[2] Invoke method response :", response)
    print('\n\n')
##########################################################################################################################################
    chat = get_chatLLM()
    
    conversation_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("Translate the following text from {input_language} to {output_language}."),
        HumanMessagePromptTemplate.from_template("{text}")
    ])
    
    # Format the prompt
    #conversation_prompt = conversation_template.invoke({"input_language":"English", "output_language":"French", "text":"I love my cat"})
    conversation_prompt = conversation_template.format(input_language="English", output_language="French", text="I love my cat")

    #print(conversation_prompt)
    #invoke_response = chat(conversation_prompt.to_string()) # .to_string() converts the prompt to a string
    invoke_response = chat(conversation_prompt)
    
    print("[3] w/o Invoke method response :", invoke_response)
    print('\n\n')

    # Example of using invoke method
    response_content = llm.invoke("Why is the sky blue?")
    print("\n[4] Invoke method response content:", response_content)

    # Example of using invoke method
    response_content = llm("Why is the sky blue?")
    print("\n[5] Direct call response content:", response_content)

    # Example of raw call to get raw response
    raw_response = llm.raw("Why is the sky blue?")  # Using the raw method
    print("\n[6] Direct call raw response:", raw_response)  # Should be the same as invoke
