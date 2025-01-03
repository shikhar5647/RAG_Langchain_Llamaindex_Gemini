# -*- coding: utf-8 -*-
"""Summarization_large_documents.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1R8DEBOz2rij_Q713-5-0UPYkNdNZ6ka3
"""

!pip install --quiet langchain

!pip install --quiet langchain-google-genai

import os
import getpass
os.environ['GOOGLE_API_KEY'] = getpass.getpass('Gemini API Key:')

!pip install langchain-community

from langchain import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document

loader = WebBaseLoader("https://blog.google/technology/ai/google-gemini-ai/#sundar-note")
docs = loader.load()

print(docs)

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                 temperature=0.7, top_p=0.85)

# To extract data from WebBaseLoader
doc_prompt = PromptTemplate.from_template("{page_content}")
llm_prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
llm_prompt = PromptTemplate.from_template(llm_prompt_template)
print(llm_prompt)

stuff_chain = (
    {
        "text": lambda docs: "\n\n".join(
            format_document(doc, doc_prompt) for doc in docs
        )
    }
    | llm_prompt         # Prompt for Gemini
    | llm                # Gemini function
    | StrOutputParser()  # output parser
)

stuff_chain.invoke(docs)

