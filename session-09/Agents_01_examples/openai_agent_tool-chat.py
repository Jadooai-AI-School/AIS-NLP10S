import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from tools import get_current_temperature, search_wikipedia, duckduckgo_news_search, serper_search

# Load environment variables from .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@cl.on_chat_start
async def start():
    # Initialize the external client
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Define the model
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    # Configure the run settings
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # Initialize the agent
    agent = Agent(
        name="KnowledgeAgent",
        instructions="""You are a knowledgeable assistant capable of fetching current temperatures,
        performing web searches, searching Wikipedia, and performing news searches.
        You reattempt tool call if some information is not available from the tool.""",
        model=model,
        tools=[get_current_temperature, search_wikipedia, duckduckgo_news_search, serper_search],
    )

    # Set up the chat session
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    cl.user_session.set("agent", agent)

    await cl.Message(content="Welcome to the KnowledgeAgent! How can I assist you today?").send()

@cl.on_message
async def main(message: cl.Message):
    # Send a placeholder message while processing
    msg = cl.Message(content="Processing your request...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    # Retrieve and update the chat history
    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})

    try:
        # Run the agent with the updated history
        result = Runner.run_sync(starting_agent=agent, input=history, run_config=config)
        response_content = result.final_output

        # Update the placeholder message with the actual response
        msg.content = response_content
        await msg.update()

        # Update the session with the new history
        cl.user_session.set("chat_history", result.to_input_list())

        # Optional: Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")

    except Exception as e:
        msg.content = f"An error occurred: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    os.system("clear")
    