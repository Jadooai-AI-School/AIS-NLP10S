# Imports - Gathering necessary libraries and modules
import json
import requests
from newspaper import Article, build
from langchain.prompts import PromptTemplate
from ais_utils.Model_from_LC_Ollama import get_LLM
import os

# Defining a function to fetch articles from URLs
def fetch_article(url, headers=None, timeout=30):
    """Fetching the content of an article from the given URL."""
    headers = headers or {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }

    try:
        with requests.Session() as session:  
            response = session.get(url, headers=headers, timeout=timeout)

        if response.status_code == 200:
            print("Successfully fetched the article. Now parsing...")
            article = Article(url)
            article.download()
            article.parse()
            return article

        else:
            print(f"Failed to fetch article at {url} (status code: {response.status_code})")
    except Exception as e:
        print(f"Encountered an error while fetching article at {url}: {e}")

# Loading the language model (LLM) for summarization tasks
print("Loading the language model...\n")
llm = get_LLM()

# Defining the URL of the article to summarize
#article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"
article_url = "https://en.wikipedia.org/wiki/Main_Page"

os.system('cls' if os.name == 'nt' else 'clear')
print(f"Fetching article from: {article_url}\n")

# Fetching the article content using the defined function
article = fetch_article(article_url)

# Displaying the original article's text
print('\nORIGINAL ARTICLE:')
print(article.text)
print('\n\nSUMMARY:\n')

if article:
    # Creating a prompt template for article summarization
    prompt_template = PromptTemplate(
        input_variables=["article_title", "article_text"],
        template="""You are a skilled assistant specializing in summarizing online articles.

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
    summary = chain.invoke({"article_title": article.title, "article_text": article.text})
    
    # Printing the generated summary
    print(summary) 
