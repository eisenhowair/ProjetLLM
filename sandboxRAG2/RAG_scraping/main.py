from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable.config import RunnableConfig
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser

import chainlit as cl

from operator import itemgetter
import re
import unicodedata
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import socket

# https://fr.wikipedia.org/wiki/Les_Trois_Mousquetaires

def clean_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def is_reachable_url(url):
    try:
        domain = re.findall(r'^(?:http|ftp)s?://([^/]+)', url)[0]
        socket.gethostbyname(domain)
        return True
    except Exception as e:
        print(f"Erreur de résolution DNS pour {url}: {e}")
        return False

def scrap_url(url):
    if not is_valid_url(url):
        print(f"L'URL '{url}' n'est pas bien formée.")
        return []

    if not is_reachable_url(url):
        print(f"L'URL '{url}' n'est pas accessible.")

        return []
    docs = []
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifie que la requête s'est bien passée
    except requests.RequestException as e:
        print(f"Erreur lors de la requête à l'URL '{url}': {e}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    text = clean_text(text)
    docs.append(Document(page_content=text, metadata={"source": url}))
    print(docs)
    return docs


llm_local = Ollama(base_url="http://localhost:11434", model="llama3:instruct")

embedding_model="hkunlp/instructor-large"
embedding = HuggingFaceEmbeddings(model_name=embedding_model)


@cl.on_chat_start
async def factory():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

    url = await cl.AskUserMessage(content="Veuillez entrer l'URL que vous souhaitez indexer :", author="Aide", timeout=3000).send()
    docs = scrap_url(url['output'])

    while(len(docs) == 0):
        await cl.Message(content=f"L'url {url['output']} est invalide", author="Aide").send()
        url = await cl.AskUserMessage(content="Veuillez entrer l'URL que vous souhaitez indexer :", author="Aide", timeout=3000).send()
        docs = scrap_url(url['output'])

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
    print("on_chat_start")


@cl.step(type="retrieval", name="invoke avec retriever")
def r_invoke(question):
    retriever = cl.user_session.get("retriever")
    res_invoke = retriever.invoke(question)
    return res_invoke


@cl.step(type="run", name="Mise en place du Runnable")
def setup_model():

    memory = cl.user_session.get("memory")

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
    res_invoke = r_invoke(message.content)
    prompt = message.content + " \nSources :"
    for r in res_invoke:
        prompt += r.page_content

    system_instructions = """
    Vous êtes un assistant français. Votre but est de répondre aux questions uniquement à l'aide des [[sources]] qu'on vous donne.
    """

    msg = cl.Message(content="")
    async for chunk in runnable_model.astream(
        {"sys_instruction": system_instructions, "prompt": prompt},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()