# Imports - Bringing in essential libraries and modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from ais_utils.Model_from_LC_Ollama import get_LLM
import faiss
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Or "true" if you want parallelism


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

# Writing the text to a local file for further processing
with open("my_file.txt", "w", encoding="utf-8") as file:  
    file.write(text)

# Loading the text data from the file using TextLoader
loader = TextLoader("my_file.txt")
documents = loader.load()

# Clearing the console for a clean output
os.system("clear")

# Displaying the number of loaded documents
print(f"Loading complete!\nNumber of documents loaded: {len(documents)}")

# Printing the content of the first document if available
if documents:
    print("\nHere's a preview of the first document's content:\n")
    print(documents[0].page_content)
    
# Splitting the document into smaller chunks
print("\nSplitting the document into smaller chunks for easier processing...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
docs = text_splitter.split_documents(documents)
print(f"Splitting complete! Number of chunks created: {len(docs)}")

# Creating embeddings for the document chunks using SentenceTransformer
print("\nGenerating numerical representations (embeddings) of each document chunk...")
embed_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
document_embeddings = embed_model.encode([doc.page_content for doc in docs])

# Creating a FAISS index for efficient similarity search
print("\nCreating a FAISS index to quickly find relevant information...")
dimension = document_embeddings[0].shape[0] 
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)
print("Index creation complete!\n")

# Loading the language model (LLM) for question-answering
print("Loading language model to generate answers...\n")
llm = get_LLM()

# Defining a custom retriever function for searching the FAISS index
def retriever(query, index, k=3):
    """Retrieves the k most similar documents to the query."""
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, k)

    # Ensure indices are valid
    valid_indices = [i for i in indices[0] if i < len(documents)] 
    return [documents[i].page_content for i in valid_indices]
#========================================================================================================
# Asking a question about PaLM
query = "What is PaLM?"
print(f"\nAsking the question: '{query}'\n")

# Retrieving relevant context from the FAISS index
context = retriever(query, index, k=3)
print("CONTEXT", context) 
context_str = "\n".join(context)
#print(context_str)

# Preparing the prompt for the LLM
prompt_template = PromptTemplate(
    input_variables=["article_title", "article_text"], 
    template="Write an article on {article_title}: {article_text}"
)

# Running the LLM chain to generate the article
prompt_input = {"article_title": "PaLM Information", "article_text": context_str} 
print("Generating an article based on the retrieved context...\n")
chain = prompt_template | llm
response = chain.invoke(prompt_input)

# Printing the generated article
print("Here's the generated article:\n")
print(response) 

