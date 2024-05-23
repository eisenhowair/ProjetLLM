import requests
from bs4 import BeautifulSoup

import time
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

# from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
import chainlit as cl
import os
import json
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

from operator import itemgetter
from PyPDF2 import PdfReader
from typing import List


urls = [
    "https://fr.wikipedia.org/wiki/Les_Trois_Mousquetaires",
]

docs = []

for url in urls:
    # Send an HTTP request to the URL of the webpage you want to access
    response = requests.get(url)


    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")
    # Extract the text content of the webpage
    text = soup.get_text()
    docs.append(Document(page_content=text, metadata={"source": url}))



llm_local = Ollama(base_url="http://localhost:11434", model="llamama")
# embedding = OllamaEmbeddings(
#     base_url="http://localhost:11434", model="nomic-embed-text"
# )

embedding_model="hkunlp/instructor-large"
embedding = HuggingFaceEmbeddings(model_name=embedding_model)


@cl.on_chat_start
async def factory():

    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))


    child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)

    vectorstore = Chroma(collection_name="full_documents", embedding_function=embedding)

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(docs, ids=None)

    
    cl.user_session.set("retriever", retriever)
   


@cl.step(type="retrieval", name="similarity search avec retriever")
def res_sim_search(question):

    retriever = cl.user_session.get("retriever")

    res_invoke = retriever.invoke(question)
    
    return res_invoke


@cl.step(type="run", name="Mise en place du Runnable")
def setup_model():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """{sys_instruction}\n\nPrompt: {prompt}""",
            ),
            MessagesPlaceholder(variable_name="history"),
        ]
    )

    runnable_exercice = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | llm_local
        | StrOutputParser()
    )
    return runnable_exercice


@cl.on_message
async def main(message):

    memory = cl.user_session.get("memory")

    runnable_model = setup_model()
    res_invoke = res_sim_search(message.content)
    prompt = message.content + " \nSources :"
    for r in res_invoke:
        prompt += r.page_content

    system_instructions = """
    Vous êtes un assistant français. Votre but est de répondre aux questions uniquement à l'aide des Sources qu'on vous donne.
    """

    msg = cl.Message(content="")
    async for chunk in runnable_model.astream(
        {"sys_instruction": system_instructions, "prompt": prompt},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()