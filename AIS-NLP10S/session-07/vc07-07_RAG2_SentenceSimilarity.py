import os
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from ais_utils.Model_from_LC_Ollama import get_chatLLM
from ais_utils.docs_load_index import load_documents, create_index, load_index_db   # newly created

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to suppress tokenizer warning during embeddings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#documents = load_documents({"pdf": "data/", 'docx', 'data/'})
#vector_dbatabase = create_index(documents, 1000, 100, "index1")
vector_database = load_index_db('my_index')

llm = get_chatLLM('llama3')

template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Give very specific but detailed answer, preferaably in bullets form.

Context:
{context}

Question: 
{question}

Answer:"""

qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain = qa_prompt | llm
os.system('clear')

# Assuming that data directory path is given by the user
def ask_question_to_book(question: str, verbose=False):
    
    matching_docs = vector_database.similarity_search(question, k=5)
    if verbose:
        print(f"Matching Documents:\n{matching_docs}\n")
    
    result = qa_chain.invoke({"context": matching_docs, "question": question}).content
    print(result, "\n")

    return result

# Ask questions
ask_question_to_book("What is Linux composed of?")
ask_question_to_book("Does Linux get Virus?")
