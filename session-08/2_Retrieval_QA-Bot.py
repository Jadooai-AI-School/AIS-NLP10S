from ais_rag.vectordb import Retriever, FAISSDB
#from ais_rag.prompt_builder import PromptBuilder
from ais_rag.chains import SimpleQAChain, RetrievalQAChain
from ais_rag.model import get_LLM
import os, pathlib
from pprint import pprint
os.chdir(pathlib.Path(__file__).parent)
print("Current working directory:", os.getcwd())
######################################################################################
os.system('cls' if os.name == 'nt' else 'clear')
######################################################################################
def main():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')  # Clearing the console
    
    # Initialize FAISS database and retriever
    faiss_db = FAISSDB(model_name='sentence-transformers/all-MiniLM-L6-v2')
    retriever = Retriever(faiss_db)
    
    # Initialize LLM
    llm = get_LLM('llama3.2')
    
    # Initialize RetrievalQAChain
    qa_chain = RetrievalQAChain(llm=llm, retriever=retriever)

    # Define a QA prompt template that emphasizes using the provided context
    qa_prompt_template = """
    You are a helpful, respectful Assistant. Use the provided context to answer the question.
    Politely respond to the greetings.

    You may use the following conversation trail ONLY to better understand the question and provide an answer, and do not echo it.
    '''conversation trail: {{history}}'''

    Context: {{context}}

    You just answer the question straightly. No unnecessary or extra words which are not part of the concise reply.
    You prefer to format your response in a bulleted list.
    
    Question: {{question}}
    
    Assistant:
    """
    qa_chain.add_qa_prompt_template(qa_prompt_template)

    #clearing screen
    os.system('cls' if os.name == 'nt' else 'clear')
    # QA Loop
    print('Start chatting with the bot (type /bye to exit)')
    while True:
        print('*'*10, '[User]', '*'*79)
        user_input = input("Enter your question: ")
        if user_input.lower() == "/bye":
            print("Goodbye!")
            break
        
        print('-'*10, '[Bot]', '-'*80)
        response = qa_chain.invoke(query=user_input)
        pprint(response)
        print('\n')

if __name__ == "__main__":
    main()
