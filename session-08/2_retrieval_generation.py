from ais_rag.vectordb import Retriever, FAISSDB
from ais_rag.prompt_builder import PromptBuilder
from ais_rag.model import get_LLM
from pathlib import Path
import os

# Clear the console screen
os.system('cls' if os.name == 'nt' else 'clear')

# Change the current working directory to the script's directory
os.chdir(Path(__file__).parent)
print("Current working directory:", os.getcwd())
######################################################################################
# Initialize the FAISS database
faiss_db = FAISSDB(model_name='sentence-transformers/all-MiniLM-L6-v2')
print("FAISS database initialized.")

# Initialize the retriever
retriever = Retriever(faiss_db)
print("Retriever initialized.")

######################################################################################
# Define the prompt template
print("Defining the prompt template...")
template = """
You are a helpful assistant, and reply politely and respectfully. 
Given the following information, answer the question.
Can arrange the longer answers in bullet form. 

Context:
'''\n{{context}}\n'''

Question: {{question}}

Answer:
"""
print("Prompt template defined.")

# Initialize the prompt builder with the retriever
prompt_builder = PromptBuilder(retriever=retriever, template_str=template)
print("PromptBuilder initialized.")

# Initialize the language model
llm = get_LLM('llama3.2')
print("Language model 'llama3.2' initialized.")

######################################################################################
# QA Loop
print("==========================================")
print("\n\nStarting QA-RAG loop. Type '/bye' to exit.")
while True:
    print('*'*10, '[User]', '*'*79)
    user_input = input("Enter your question: ")
    if user_input.lower() == "/bye":
        print("Goodbye!")
        break
    
    print('-'*10, '[Bot]', '-'*80)
    # Build the prompt
    prompt = prompt_builder.build(query=user_input)
    print("Top K Chunks:\n", prompt)
    response = llm.generate(prompt=prompt)
    print("LLM Response:\n", response)
    print('\n')
    
    