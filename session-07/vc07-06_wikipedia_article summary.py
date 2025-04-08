# Imports - Bringing in necessary modules
import os
from langchain_community.document_loaders import WikipediaLoader
from langchain.prompts import PromptTemplate
from ais_utils.Model_from_LC_Ollama import get_LLM

# Loading the language model (LLM) for summarization tasks
print("Loading the language model...\n")
llm = get_LLM()

# Setting the Wikipedia search term (you can change this!)
search_term = "Large language model"

# Clearing the console
os.system('cls' if os.name == 'nt' else 'clear')
print(f"Fetching Wikipedia article for: {search_term}\n")

# Loading the Wikipedia article using LangChain's built-in loader
loader = WikipediaLoader(query=search_term, lang="en", load_max_docs=1)
documents = loader.load()

# Displaying the original article content
if documents:
    article_text = documents[0].page_content
    article_title = documents[0].metadata.get("title", search_term)

    print('\nORIGINAL ARTICLE:\n')
    print(article_text)
    print('\n\nSUMMARY:\n')

    # Creating a summarization prompt template
    prompt_template = PromptTemplate(
        input_variables=["article_title", "article_text"],
        template="""You are a skilled assistant specializing in summarizing Wikipedia articles.

        Here's the article you want to summarize.

        ==================
        Title: {article_title}

        {article_text}
        ==================

        Please provide a concise and informative summary of the article above.
        """
    )

    # Generating the summary using the LLM chain
    print("Generating summary...\n")
    chain = prompt_template | llm
    summary = chain.invoke({"article_title": article_title, "article_text": article_text})

    # Printing the generated summary
    print(summary)
else:
    print("No content found for the given Wikipedia query.")
