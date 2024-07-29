import json
import requests
from newspaper import Article, build
from langchain_core.prompts import PromptTemplate
#from langchain_community.llms.ollama import Ollama
from models.Model_from_LC_Ollama import get_LLM
import os

# Fetch the Article
def fetch_article(url, headers=None, timeout=30):
    """Fetches the content of an article from a given URL."""
    headers = headers or {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }

    try:
        with requests.Session() as session:  # Better practice: Use context manager for sessions
            response = session.get(url, headers=headers, timeout=timeout)

        if response.status_code == 200:
            article = Article(url)
            article.download()
            article.parse()
            return article

        else:
            print(f"Failed to fetch article at {url} (status code: {response.status_code})")
    except Exception as e:
        print(f"Error occurred while fetching article at {url}: {e}")


# LLM Setup
#llm = Ollama(model="llama3")  # Using langchain_community library
llm = get_LLM()
# Article URL
#article_url = "#https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"
article_url = "https://edition.cnn.com/world"

os.system('cls' if os.name == 'nt' else 'clear')
    
# Fetch Article
article = fetch_article(article_url)
print('ARTICLE:')
print(article.text)
print('\n\nSUMMARY:')

if article:
    # Prompt Template
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

    # Generate Summary
    chain = prompt_template | llm
    summary = chain.invoke({"article_title": article.title, "article_text": article.text})
    
    print(summary)  # Access the 'text' key for the Ollama response

