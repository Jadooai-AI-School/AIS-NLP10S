import uuid
import threading
from typing import Iterator

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from ais_utils.Model_from_LC_Ollama import get_chatLLM
###########################################################################################################
store = {}
HISTORY_LIMIT = 4
llm = get_chatLLM()
SYSTEM_MESSAGE = "You are a wise helpful assistant. Your name is Aliza you will address the user as Charlie. Always answer in that all times."
SESSION_ID = uuid.uuid4()  # initial session id for memory store
###########################################################################################################
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)
###########################################################################################################
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of limited chat message history storing only the last K messages."""

    messages: list[BaseMessage] = Field(default_factory=list)
    max_messages: int = 10

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
        self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        self.messages = []
###########################################################################################################
def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory(max_messages=HISTORY_LIMIT)
    return store[session_id]
###########################################################################################################
llm_chain = prompt_template | llm

llm_with_memory = RunnableWithMessageHistory(
    llm_chain,
    get_history,
    input_messages_key="query",
    history_messages_key="history",
)
###########################################################################################################
def chat(query: str) -> None | Iterator[str]:
    response = llm_with_memory.invoke(
        {"query": query}, config={"configurable": {"session_id": SESSION_ID}}
    )
    print(response.content)

###########################################################################################################
if __name__ == "__main__":
    import os
    os.system("clear")
    try:
        while True:
            query = input("You: ")
            if query.lower() == "quit":
                break
            chat(query)
            print("\n")
    except KeyboardInterrupt:
        print("Shutting down...")