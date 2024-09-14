#!/usr/bin/env python
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, field_validator
from ais_utils.Model_from_LC_Ollama import get_chatLLM, get_LLM
#####################################################################################
import os
os.system("clear")

#define field types and validators
class PersonInfo(BaseModel):
    name: str = Field(description="Person's name")
    age:  int = Field(description="Person's age")
    city: str = Field(description="Person's city of residence")

    @field_validator('age')
    def age_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError("Age must be positive")
        return value

# 1. Create prompt template
prompt_template = PromptTemplate.from_template(
    """Extract information as per given json format from the following text: {text}.

    If the information is not present, indicate it as 'Not found'.

    The text can be in any language, you should translate to English first then extract the information.

    Your answer should be valid JSON and should follow this format:

    example json_format:
    {{"name:" "Joe", "age": 40, "city": "SFO"}}
    """
)

# 2. Create model
model = get_LLM('gemma2:2b')

# 3. Create parser
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 4. Create chain
chain = prompt_template | model | parser

# 5. Make Inference
texts = [
    "My name is Alice, I am 30 years old and I live in New York City.",
    "Je m'appelle Pierre, j'ai 25 ans et j'habite à Paris.",
    "Hola, me llamo Ana. Tengo 22 años y vivo en Madrid."
]
for text in texts:
    try:
        response = chain.invoke({"text": text})
        print("Response from chain (structured extraction):", response)
    except Exception as e:
        print(f"Error extracting information: {e}")
