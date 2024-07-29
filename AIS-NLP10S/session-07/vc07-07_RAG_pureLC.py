from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ais_utils.Model_from_LC_Ollama import get_LLM
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to suppress tokenizer warning during embeddings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # for TF Warning

# Load and preprocess the PDF document
loader = PDFPlumberLoader("data/LINUX Easy Linux For Beginners.pdf")
data: list[Document] = loader.load_and_split()

# Extract texts from the pages
page_texts: list[str] = [page.page_content for page in data]
print(page_texts[6][:1000])

# Instantiate Hugging Face embeddings
embeddings_api = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},  # Use GPU if available
)

# chunking to be made
#
# Create the FAISS vector store from the preprocessed texts
vector_database = FAISS.from_texts(page_texts, embeddings_api)

# Create a retriever from the vector store
retriever = vector_database.as_retriever()

# Instantiate the LLM
llm = get_LLM()

# Define the prompt template for question answering
template = """Use ONLY the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: 
{question}

Answer:"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Define a function to handle the question answering process
def answer_question(question: str):
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question, k=5)
    # Concatenate the retrieved documents to form the context
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    # Format the prompt using the template
    #formatted_prompt = prompt_template.format(context=context, question=question)
    # Get the LLM's answer
    #response = llm.generate(formatted_prompt)
    chain = prompt_template | llm
    
    return chain.invoke({"context": context, "question": question})


import os
os.system('clear')

# Ask a question and get the answer
response = answer_question("What is Linux composed of?")
print(response)

# Additional questions
# response = answer_question("What is Virus threat to Linux?")
# print(response)
# response = answer_question('Linux as Compared to other Operating Systems')
# print(response)
