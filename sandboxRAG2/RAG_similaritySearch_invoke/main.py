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

from operator import itemgetter
from PyPDF2 import PdfReader
from typing import List


llm_local = Ollama(base_url="http://localhost:11434", model="llama3:8b")
embedding = OllamaEmbeddings(
    base_url="http://localhost:11434", model="nomic-embed-text"
)


start = time.time()  #

# Load blog post
loader = TextLoader("./liam.txt")
data = loader.load()
loader = TextLoader("./emma.txt")
data2 = loader.load()
loader = TextLoader("./ammamellen.txt")
data3 = loader.load()

docs = data + data2 + data3


end = time.time()  #
timee = end - start  #
print("Chargement de tous les documents=", timee)  #

# PDFLoader
# import PyPDF2

# pdf = PyPDF2.PdfReader("liam.pdf")
# pdf_text = ""
# for page in pdf.pages:
#     pdf_text += page.extract_text()

# pdf = PyPDF2.PdfReader("liam.pdf")
# pdf_text2 = ""
# for page in pdf.pages:
#     pdf_text2 += page.extract_text()

# docs = pdf_text + pdf_text2


@cl.on_chat_start
async def factory():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

    ####SIMILARITY_SEARCH
    start = time.time()  #

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

    end = time.time()  #
    timee = end - start  #
    print("embedding similarity_search=", timee)  #
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("vectorstore", vectorstore)


@cl.step(type="retrieval", name="similarity search avec retriever")
def res_sim_search(question):

    vectorstore = cl.user_session.get("vectorstore")
    retriever = cl.user_session.get("retriever")
    start = time.time()  #
    res_similarity_search = vectorstore.similarity_search(question)
    end = time.time()
    final_time = end - start
    print("similarity_search()=", final_time)  #

    start = time.time()  #

    sources = set()

    for block in res_similarity_search:
        sources.add(block.metadata["source"])

    print("Sources :")
    for b in sources:
        print(b)

    end = time.time()  #
    timee = end - start  #
    print("Chargement des documents pertinents=", timee)  #

    # load relevant documents
    docs = []
    for source in sources:
        loader = TextLoader(source)
        data = loader.load()
        docs.extend(data)

    start = time.time()
    res_invoke = retriever.invoke(question)

    end = time.time()  #
    timee = end - start  #
    print("invoke()=", timee)  #

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

    # setup_model() et trouve_contexte() à adapter suivant ce qui est recherché
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

    # memory.chat_memory.add_user_message(message.content)
    # memory.chat_memory.add_ai_message(msg.content)
    await msg.send()


"""
async def old_main(message):

    prompt_template = PromptTemplate(
        template="{instructions}\n\nPrompt: {prompt}",
        input_variables=["instructions", "prompt"],
    )

    prompt = "Comment s'appelle le fils de la servante ? Sources :"
    for r in res_invoke:
        prompt += r.page_content

    full_prompt = prompt_template.format(
        instructions=system_instructions, prompt=prompt
    )
    response = llm_local.generate([full_prompt])

    end = time.time()  #
    timee = end - start  #
    print("Chargement modèle et génération réponse=", timee)  #

    print(response.generations[0][0].text)
"""
