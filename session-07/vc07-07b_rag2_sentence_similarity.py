# Imports - Gathering necessary libraries and modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disabling tokenizer parallelism to avoid warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppressing TensorFlow warning messages
import warnings
warnings.filterwarnings("ignore")

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from ais_utils.Model_from_LC_Ollama import get_chatLLM
from ais_utils.docs_load_index import load_documents, create_index, load_index_db

db_path = './data/index.db'
index_file = 'my_index'
#if not os.path.exists(db_path):
#    create_index(load_documents(), 1000, 100, 'my_index', persist=True)

os.system('clear')
# Loading a pre-built vectorstore index for efficient document search
print("\nLoading the pre-built vectorstore index from 'my_index'...")

vector_database = load_index_db(index_file)
if vector_database is None:
    print("Index not found. Please create an index first.")
    exit()
print("Index loading complete!\n")

# Loading the language model (LLM) for question-answering
print("Loading the language model...\n")
llm = get_chatLLM('gemma2:2b')

# Crafting a prompt template for question-answering
template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Give very specific but detailed answer, preferaably in bullets form.

Context:
{context}

Question: 
{question}

Answer:"""

qa_prompt_template = PromptTemplate.from_template(template=template)
qa_chain = qa_prompt_template | llm

# Clearing the console for a clean output
print("Ready to answer questions about your documents!\n")

# Defining a function to ask questions and retrieve answers
def ask_question_to_book(question: str, verbose=False):
    """Retrieves relevant documents, formulates a prompt, and generates answers using the LLM."""
    print(f"Searching for information related to: '{question}'...\n")
    matching_docs = vector_database.similarity_search(query=question, k=3)

    if verbose:
        print("Here are the most relevant document snippets found:\n")
        print(matching_docs)

    print("\nFormulating a response based on the retrieved context...\n")
    result = qa_chain.invoke({"context": matching_docs, "question": question}).content
    print("Answer:")
    print(result, "\n")

    return result

# Asking questions
print('LLM Responses:')
print(80*'-')
ask_question_to_book("What is Linux composed of?")
print(80*'-')
ask_question_to_book("Does Linux get Virus?")
print(80*'-')
print('Finished...')
