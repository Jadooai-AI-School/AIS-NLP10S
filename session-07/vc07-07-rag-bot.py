import uuid
import os
import threading
from typing import Iterator, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ais_utils.Model_from_LC_Ollama import get_LLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to suppress tokenizer warning during embeddings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for TF Warning

###########################################################################################################
# InMemoryHistory class and get_history function

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of limited chat message history storing only the last K messages."""

    messages: list[BaseMessage] = Field(default_factory=list)
    max_messages: int = 10

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
        self.messages = self.messages[-self.max_messages:]

    def clear(self) -> None:
        self.messages = []

def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory(max_messages=HISTORY_LIMIT)
    return store[session_id]

###########################################################################################################
# Constants and initializations
store = {}
HISTORY_LIMIT = 4
llm = get_LLM()
SYSTEM_MESSAGE = "You are a wise helpful assistant. If you are given a context, you should use it to answer the question. Your name is Aliza you will address the user as Charlie. Always answer in that all times."
SESSION_ID = uuid.uuid4()  # initial session id for memory store
###########################################################################################################
# Load and preprocess the PDF document
loader = PyMuPDFLoader("data/LINUX Easy Linux For Beginners.pdf")
data: List[Document] = loader.load_and_split()

# Extracting texts from the pages
page_texts: List[str] = [page.page_content for page in data]
print(page_texts[6][:1000])

# Instantiate Hugging Face embeddings
embeddings_api = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},  # Use GPU if available
)

# Creating the FAISS vector store from the preprocessed texts
vector_database = FAISS.from_texts(page_texts, embeddings_api)

# Creating a retriever from the vector store
retriever = vector_database.as_retriever()

# Define the prompt template for question answering
template = """Use the following pieces of 'context' to answer the Charlie's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: 
{question}

Answer:"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)

llm_chain = prompt_template | llm

llm_with_memory = RunnableWithMessageHistory(
    llm_chain,
    get_history,
    input_messages_key="query",
    history_messages_key="history",
)

###########################################################################################################
# Updated answer_question function with memory

def answer_question(question: str) -> None | Iterator[str]:
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question, k=5)
    # Concatenate the retrieved documents to form the context
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    response = llm_with_memory.invoke(
        {"query": question, "context": context}, config={"configurable": {"session_id": SESSION_ID}}
    )
    print(response)
    return response

###########################################################################################################
# Chat interface

if __name__ == "__main__":
    os.system("clear")
    try:
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            answer_question(query)
            print("\n")
    except KeyboardInterrupt:
        print("Shutting down...")
