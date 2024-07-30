from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ais_utils.Model_from_LC_Ollama import get_chatLLM

llm = get_chatLLM()

SYSTEM_MESSAGE = "You are a helpful assistant. You respond in a compassionate and professional manner."

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        ("human", "{query}"),
    ]
)

chain = prompt_template | llm | StrOutputParser()


def chat(query: str) -> None:
    for chunk in chain.stream(query):
        print(chunk, end="")


if __name__ == "__main__":
    import os
    os.system("clear")
    while True:
        query = input("You: ")
        chat(query)
        print("\n")