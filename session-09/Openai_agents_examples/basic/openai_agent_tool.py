from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

import os
from pathlib import Path
from tools import get_current_temperature, search_wikipedia, duckduckgo_news_search, serper_search

#setting current working directory
os.chdir(Path(__file__).parent)
##################################################################################################
gemini_api_key = os.getenv("GEMINI_API_KEY")
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)
##################################################################################################
agent: Agent = Agent(
    name="KnowledgeAgent",
    instructions="""You are a knowledgeable assistant capable of fetching current temperatures,\
    performing web searches, searching Wikipedia, and performing news searches.\
    You reattemmpt tool call if some information is not available from the tool.""",
    model=model,
    tools=[get_current_temperature, search_wikipedia, duckduckgo_news_search, serper_search],
    )

query = """what is weather at San Fransisco, today at 4am. Get lat/Long using websearch."""
query = """what is local news at San Fransisco, today?"""
query = """what is Rome news today?"""


result = Runner.run_sync(agent, query, run_config=config)

print("\nCALLING AGENT\n")
print(result.final_output)