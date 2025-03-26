from typing import List
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain.schema.output_parser import OutputParserException
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import re
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_API_TOKEN')
REPO_ID = os.getenv('REPO_ID')

# Instructions
instructions = """
For the given input determine if the movie review is demonstrating a positive sentiment by returning True or False:
"""

# Template
template = """Input: {input}

Instructions: {instructions}

{format_instructions}
Response: """

class GenText(BaseModel):
    review_sentiment: List[bool] = Field(description="Boolean value reflecting positive movie review sentiment if true else negative if false.")

class LangchainLLM:
    def __init__(self, temperature:float=0.1, repo_id:str=REPO_ID, hf_api_token:str=HUGGINGFACEHUB_API_TOKEN):
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=temperature,
            huggingfacehub_api_token=hf_api_token,
            task="text-generation",
        )

        self.output_parser = PydanticOutputParser(pydantic_object=GenText)
        format_instructions = self.output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "input",
                "instructions",
            ],
            partial_variables={"format_instructions":format_instructions},
        )


        self.prompt_chain = prompt | self.llm | self.output_parser
        self.response_chain = prompt | self.llm

    
    def get_response(self, text:str):
        invoke_dict = {
            "input":text,
            "instructions":instructions,
        }

        return self.prompt_chain.invoke(invoke_dict)