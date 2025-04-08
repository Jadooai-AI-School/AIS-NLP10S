from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
my_api_key = os.getenv('GROQ_API_KEY') #or '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'
#print(my_api_key)
def get_chatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", 
    base_url="https://api.groq.com/openai/v1", 
    temperature=0.2, 
    api_key=my_api_key, 
    kwargs={}
):
    chat_openai = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        model_kwargs=kwargs,
    )
    return chat_openai


if __name__=='__main__':    

    llm = get_chatGroq()
    resp = llm.invoke('What do you know about Abraham Lincoln')
    print(resp.content)