# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
# Imports - Bringing in necessary libraries and modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # Using LangChain's HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from ais_utils.Model_from_LC_Ollama import get_LLM
# ... (rest of your imports)

# Preparing text data about Google's PaLM model

text = """Google's Move to Challenge OpenAI and GPT-3
Google has opened up access to its advanced AI language model, PaLM (Pathways Language Model), to developers through an API. This move is seen as a challenge to OpenAI's GPT-3 and other large language models.

Key Points:
PaLM API: Google has launched an API (Application Programming Interface) for PaLM, allowing developers to integrate it into their applications and services.
AI Enterprise Tools: Alongside the API, Google has introduced various AI tools designed for businesses. These tools aim to simplify the generation of various content formats (text, images, code, videos, audio) using natural language prompts.
Large Language Model (LLM):** PaLM is a type of LLM, similar to OpenAI's GPT models and Meta's LLaMA. These models are trained on massive datasets and can perform a wide range of text-based tasks.
PaLM's Flexibility: Like other LLMs, PaLM is highly versatile and can be trained for various applications, such as:
    - Conversational chatbots (similar to ChatGPT)
    - Text summarization
    - Code generation
    - Other creative and informative tasks
Integration with Workspace: Google has also announced the integration of similar AI features into its Workspace applications, including Google Docs and Gmail.

Implications:

Increased Competition: Google's entry into the LLM market with PaLM intensifies competition in the AI landscape. This could lead to more innovation, faster development, and potentially lower costs for businesses and consumers.
Democratization of AI: By making PaLM accessible through an API, Google is enabling a wider range of developers and businesses to utilize powerful AI language capabilities. This could lead to new applications and services that were previously not possible.
Ethical Considerations: As with any AI technology, there are potential ethical concerns, such as the misuse of generated content or the spread of misinformation. Responsible development and usage will be crucial as these models become more prevalent.
""".replace("\n", " ")
##################################################################################################

# Writing the text to a local file named 'my_file.txt'
with open("my_file.txt", "w", encoding="utf-8") as file: 
    file.write(text)

# Loading the text data from the local file using TextLoader
loader = TextLoader("my_file.txt")
documents = loader.load()  

os.system("clear")  # Clearing the console for a clean output

# Displaying information about the loaded documents
print("\nLoading Complete! Number of documents loaded:", len(documents))  
if documents:
    print("\nHere's a preview of the first document:")
    print(documents[0].page_content)

# Splitting the loaded documents into smaller chunks 
print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Number of chunks created: {len(docs)}\n")

# Creating embeddings for the document chunks using HuggingFaceEmbeddings
print("Creating embeddings for each document chunk...")
#embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")  # Using LangChain's version
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Building a FAISS vectorstore to store the embeddings for efficient similarity search
print("\nBuilding the FAISS vectorstore...")
db = FAISS.from_documents(docs, embeddings) # Pass document chunks directly to FAISS
print("Vectorstore creation complete!\n")

# Loading the language model (LLM) for question-answering
print("Loading the Language Model (LLM)...\n")
llm = get_LLM()

# Constructing a RetrievalQA chain combining the vectorstore retriever and the LLM
print("Setting up the Question-Answering Chain...")
retriever = db.as_retriever()
qa_chain = retriever | llm
print("Ready to answer questions!\n")
# Asking a question about PaLM and retrieving the answer using the QA chain
query = "What is PaLM?"
response = qa_chain.invoke({"query": query})
print("Q:", query)
print("A:", response['result'], "\n")  # Printing the answer with a label
