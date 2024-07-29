default_model = "gemma:2b"
###########################################################################################################
# This is a Langchain LLM wrapper for the Ollama model
from langchain_community.llms import Ollama
def get_LLM(model=default_model, temperature=0.2, top_k = 40, top_p = 0.95, num_ctx = 1024, base_url = "http://localhost:11434"):
    llm = Ollama(
        base_url = "http://localhost:11434",
        model = model,
        temperature=temperature,
        top_k = top_k,
        top_p = top_p,
        num_ctx= num_ctx,
    )
    return llm

###########################################################################################################
#This is a Langchain ChatLLM wrapper for the Ollama model
from langchain_community.chat_models.ollama import ChatOllama
def get_chatLLM(model=default_model, base_url = "http://localhost:11434", temperature=0.2, top_k = 40, top_p = 0.95, num_ctx = 1024):
    chat_llm = ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_k = top_k,
        top_p = top_p,
        num_ctx= num_ctx,
    )
    return chat_llm

###########################################################################################################
from langchain_openai import ChatOpenAI
def get_chatOpenAI(model=default_model, base_url = "http://localhost:11434", temperature=0.2, kwargs = {}):
    #model_kwargs = {"top_k": top_k, "top_p": top_p, "num_ctx": num_ctx}
    chat_openai = ChatOpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key= 'ollama', # required, but unused
        model= model,
        temperature=temperature,
        model_kwargs=kwargs,
    )
    return chat_openai
###########################################################################################################
import requests
import json

URL = "http://localhost:11434/api/generate"
HEADERS = {"Content-type": "application/json"}

def preload_model(model_name: str = default_model) -> bool:
    print(f"Running model preloader for {model_name}...")
    data = {"model": model_name, "keep_alive": "10m"}

    response = requests.post(URL, data=json.dumps(data), headers=HEADERS)

    return True if response.status_code == 200 else False
###########################################################################################################
# Example usage
if __name__ == "__main__":
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    import os
    os.system("clear")
    
    llm = get_chatLLM()
    question_template = PromptTemplate(
        input_variables=["question"],
        template="{question}"
        )
    prompt = question_template.format(question="Why is the sky blue?")
    response = llm.invoke(prompt).content
    print("Invoke method response :", response) 
    print('\n\n')
    
    
    translation_template = PromptTemplate(
        input_variables=["text", "input_language", "output_language"],
        template="""
        Translate the following text from {input_language} to {output_language}.
        Text: {text}
        """)
    # Format the prompt
    prompt = translation_template.format(input_language="English", output_language="French", text="I love my cat")
    response = llm.invoke(prompt).content
    print("Invoke method response :", response)
    print('\n\n')
##########################################################################################################################################
    chat = get_chatLLM()
    
    conversation_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("Translate the following text from {input_language} to {output_language}."),
        HumanMessagePromptTemplate.from_template("{text}")
    ])
    
    # Format the prompt
    conversation_prompt = conversation_template.format_messages(input_language="English", output_language="French", text="I love my cat")
    
    invoke_response = chat.invoke(conversation_prompt).content
    print("Invoke method response :", invoke_response)
    