from ais_rag.vectordb import Retriever, FAISSDB
from ais_rag.prompt_builder import PromptBuilder
from ais_rag.model import get_LLM
import os
os.system('cls' if os.name == 'nt' else 'clear')
######################################################################################
faiss_db = FAISSDB(model_name='sentence-transformers/all-MiniLM-L6-v2')
retriever = Retriever(faiss_db)
######################################################################################
# Define the template
template = """
You are a helpful assistant, and reply politely and respectfully. 
Given the following information, answer the question.
Can arrange the longer answers in bullet form. 

Context:
'''\n{{context}}\n'''

Question: {{question}}

Answer:
"""
# Initializ the prompt builder with the retriever
prompt_builder = PromptBuilder(retriever=retriever)
llm = get_LLM()
os.system('cls' if os.name == 'nt' else 'clear')
######################################################################################
# QA Loop
while True:
    user_input = input("Enter your question (or type /bye to quit): ")
    if user_input.lower() == "/bye":
        print("Goodbye!")
        break

    # Build the prompt
    prompt = prompt_builder.build(query=user_input, template=template)
    
    # Output the prompt and get the response from the LLM
    print("Generated Prompt:\n", prompt)
    response = llm.generate(prompt=prompt)
    
    print("LLM Response:\n", response)