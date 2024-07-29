#!/usr/bin/env python
from click import prompt
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Model_from_LC_Ollama import get_chatLLM

# 1. Create prompt template
system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
model = get_chatLLM()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

x = chain.invoke({"text": "I love my cat", "language": "French"})
print(x)

##################################################################################################################

# 1. Create prompt template
template = "Translate the following text:\n'{text}' into French:"

prompt_template = PromptTemplate.from_template(template=template)#, input_variables=["text"])
   
# 2. Create model
model = get_chatLLM()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

resp = chain.invoke({'text': "I love my cat"})
print('Response from chain:', resp)