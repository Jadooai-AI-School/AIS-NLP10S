#!/usr/bin/env python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ais_utils.Model_from_LC_Ollama import get_chatLLM
import os
##################################################################################################################
os.system("clear")  # screen clear

# 1. Create prompt template
prompt_template = PromptTemplate.from_template(
    template="Translate the following text:\n'{text}' into French:"
    )
print('1. Prompt:', prompt_template)
   
# 2. Create model
model = get_chatLLM()

# 3. Create parser 
parser = StrOutputParser()  # string output parser

# 4. Create chain
chain = prompt_template | model | parser

resp = chain.invoke({'text': "I love my cat"})

print('Translation\n:', resp)
print('\n\n')
##################################################################################################################
# 1. Create Chat prompt template

system_template = "Translate the following into {language}:"

chat_prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
    ])

# 2. Create model
model = get_chatLLM()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = chat_prompt_template | model | parser

print("2. Chat Prompt:", chat_prompt_template)
resp = chain.invoke({"text": "I love my cat", "language": "French"})
print('Translation with Chat:\n', resp)
print('\n\n')
