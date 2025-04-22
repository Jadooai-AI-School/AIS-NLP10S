import os
from pathlib import Path
os.chdir(Path.absolute)



from tools import get_current_temperature, search_wikipedia, duckduckgo_news_search

from agents import Agent

agent = Agent(
    name="KnowledgeAgent",
    instructions="You are a knowledgeable assistant capable of fetching current temperatures, searching Wikipedia, and performing news searches.",
    tools=[get_current_temperature, search_wikipedia, duckduckgo_news_search],
)
