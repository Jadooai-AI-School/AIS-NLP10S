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
import sys, os
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    #!pip install -qqq ctransformers torch
    os.system("pip install -qqq ctransformers torch")
def get_ColabLLM(temperature=0.2, top_k = 40, top_p = 0.95, num_ctx = 1024):
    if not IN_COLAB:
        raise Exception("Not running in Colab!")
    
    from ctransformers import AutoModelForCausalLM
    import torch

    """1(c). Initialize settings"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    if device != "cuda":
        raise Exception("Colab GPU not available!")
    
    dtype = torch.float16 if device == "cuda" else torch.float32

    """2(a). Defining Model location"""

    # select model from https://huggingface.co/models
    # huggingface path (can also use locally downloaded model-path)
    model_name = "TheBloke/zephyr-7B-beta-GGUF"
    model_file = "zephyr-7b-beta.Q4_K_M.gguf"
    model_type="mistral"

    """2(b). Instantiating Model object"""

    # download model from huggingface
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        model_file=model_file,
        model_type=model_type,
        gpu_layers=50,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        context_length=num_ctx,
        )
    return llm


llm = get_ColabLLM()
response = llm.invoke("Why is the sky blue?").content
print(response)

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
    
    