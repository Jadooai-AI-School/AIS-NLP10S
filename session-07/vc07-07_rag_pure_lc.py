# Imports - Gathering necessary libraries and modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ais_utils.Model_from_LC_Ollama import get_LLM
from pathlib import Path
os.system('clear')
os.chdir(Path(__file__).resolve().parent)  # Setting the working directory to the script's location
data_path = Path("data")  # Defining the data path

# Loading and preprocessing the PDF document
print("\nLoading PDF document using PDFPlumber...")

loader = PyPDFLoader(data_path/"LINUX Easy Linux For Beginners.pdf")
data = loader.load_and_split()
print("PDF loading and splitting complete!\n")

# Extracting text content from each page of the document
page_texts = [page.page_content for page in data]
print("Displaying a snippet of extracted text from page 6:")
print(page_texts[6][:1000], "\n")  # Show first 1000 characters

# Instantiating Hugging Face embeddings for text representation
print("Creating embeddings for each text chunk...")
embeddings_api = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={"device": "cpu"})

# Creating a FAISS vector store for efficient similarity search
print("\nBuilding a FAISS vectorstore to index the document...")
vector_database = FAISS.from_texts(page_texts, embeddings_api)
print("Vectorstore creation complete!\n")

# Creating a retriever object from the vector store
retriever = vector_database.as_retriever()

# Loading the language model (LLM) for question-answering
print("Loading language model to generate answers...\n")
llm = get_LLM()

# Defining a prompt template for question answering
template = """Use ONLY the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: 
{question}

Answer:"""

prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

# Defining a function to handle the question answering process
def answer_question(question: str):
    """Retrieves relevant context and generates an answer using the LLM."""
    print("Retrieving relevant context from the vectorstore...")
    retrieved_docs = retriever.invoke(question, k=5)

    print("Combining retrieved context...")
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Create the chain for question answering
    chain = prompt_template | llm

    print("Generating answer...\n")  # Indicate answer generation
    return chain.invoke({"context": context, "question": question})

# Clearing the console for output clarity
os.system('clear')
print("Ready to answer questions!\n")

# Ask a question and get the answer
response = answer_question("What is Linux composed of?")
print("Q:", "What is Linux composed of?")
print("A:", response, "\n") 

# Example Additional questions (uncomment to use)
# response = answer_question("What is Virus threat to Linux?")
# print("Q:", "What is Virus threat to Linux?")
# print("A:", response, "\n")
# response = answer_question("Linux as Compared to other Operating Systems")
# print("Q:", "Linux as Compared to other Operating Systems")
# print("A:", response, "\n")
