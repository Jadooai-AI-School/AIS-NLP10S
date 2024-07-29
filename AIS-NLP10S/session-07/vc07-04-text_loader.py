from langchain_community.document_loaders import TextLoader
import os

from torch import embedding
from zmq import device

# text to write to a local file
# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
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
"""
from langchain_community.document_loaders import TextLoader

# ... (Your text variable remains unchanged) ...

# Write text to local file
with open("my_file.txt", "w", encoding="utf-8") as file:  # Add encoding for better handling
    file.write(text)

# Use TextLoader to load text from local file
loader = TextLoader("my_file.txt")
documents = loader.load()  # Use the more descriptive variable name 'documents'

os.system("clear")
# Print the number of documents (should be 1 in this case)
print(f"Number of documents loaded: {len(documents)}")

# Print the content of the first (and only) document
if documents:
    print("Content of the first document:")
    print(documents[0].page_content)
    
    
from langchain.text_splitter import CharacterTextSplitter

# create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)

# split documents into chunks
docs = text_splitter.split_documents(documents)

print(len(docs))

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
#from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain_openai.llms import OpenAI

from langchain_community.vectorstores import FAISS  
from models.Model_from_LC_Ollama import get_LLM
# Create Sentence Transformers embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={"device": "cpu"})
# Create a vectorstore from documents
db = FAISS.from_documents(documents, embeddings)

# Define your LLM 
llm = get_LLM()

# Create the RetrievalQA chain
retriever = db.as_retriever()
qa_chain = retriever | llm

# Ask a question
query = "What is PaLM?"
response = qa_chain.invoke(query)
print(response)


