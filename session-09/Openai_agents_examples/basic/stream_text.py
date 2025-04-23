import asyncio

from openai.types.responses import ResponseTextDeltaEvent

from agents import Agent, Runner
from _get_model import GeminiModel
model, config = GeminiModel()


async def main():
    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
        model=model,
    )

    result = Runner.run_streamed(agent, input="Please tell me 5 jokes.", run_config=config)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
