import os
import chainlit as cl
from numpy import vectorize
from utils.manip_documents import *

from langchain_community.vectorstores import FAISS
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

from operator import itemgetter
from typing import List


from chainlit.input_widget import TextInput, Select


@cl.password_auth_callback
def auth_callback(username: str, password: str):

    if (username, password) == ("elias", "elias"):
        return cl.User(
            identifier="Elias", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.step(type="run", name="Mise en place du Runnable")
def setup_model():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    prompt_exercice = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Instruction: Répondre en francais à la question de l'utilisateur en te basant **uniquement** sur le contexte suivant fourni.
                Si tu ne trouves pas la réponse dans le contexte, demande à l'utilisateur d'être plus précis au lieu de deviner.
                Context:{context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Question: {question}"),
            ("ai", """Réponse:"""),
        ]
    )

    runnable_exercice = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt_exercice
        | model
        | StrOutputParser()
    )
    return runnable_exercice


@cl.step(type="retrieval", name="Context via similarity_search")
def trouve_contexte(question):
    retriever = cl.user_session.get("retriever")
    search_results = retriever.vectorstore.similarity_search(question, k=10)

    # Utiliser un dictionnaire pour regrouper les chunks par source
    results_by_source = {}
    for result in search_results:
        source = result.metadata["source"]
        if source not in results_by_source:
            results_by_source[source] = []
        results_by_source[source].append(result)

    # sources différentes
    relevant_sources = list(results_by_source.keys())[:2]
    # chunks par source
    relevant_results = [results_by_source[source][:10] for source in relevant_sources]

    # Aplatir la liste des résultats
    relevant_results = [chunk for sublist in relevant_results for chunk in sublist]

    filenames = [result.metadata["source"] for result in relevant_results]
    short_filenames = [os.path.basename(file) for file in filenames]
    print("modèle d'embedding:" + str(embeddings))
    print("Files used for context:", short_filenames)
    context = "\n".join(
        [
            f"-----\nSource: {os.path.basename(result.metadata['source'])}\n{result.page_content}"
            for result in relevant_results
        ]
    )

    print("-------------------\n" + context)
    return context


@cl.on_chat_start
async def factory():
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="addDocuments",
                label="Précisez le chemin",
                initial="differents_textes/...",
            ),
            Select(
                id="model",
                label="Model",
                values=[
                    "instructor-xl",
                    "instructor-base",
                    "instructor-large",
                    "mpnet-v2",
                    "camembert-base",
                ],
                initial_index=0,
            ),
        ]
    ).send()
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

    charge_index(index_path)


def charge_index(index_path):
    print(f"index_path: {index_path}\nembeddings:{embeddings}")
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            index_path, embeddings=embeddings, allow_dangerous_deserialization=True
        )
        print("Index chargé à partir du chemin existant.")
    else:
        chunks = load_new_documents("differents_textes")

        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

        vectorstore.save_local(index_path)
        print("Nouvel index créé et sauvegardé.")

    retriever = vectorstore.as_retriever()
    cl.user_session.set("retriever", retriever)


@cl.on_message
async def main(message):
    memory = cl.user_session.get("memory")
    question = message.content
    print("Question:" + question)

    # setup_model() et trouve_contexte() à adapter suivant ce qui est recherché
    runnable_model = setup_model()
    msg = cl.Message(content="")
    async for chunk in runnable_model.astream(
        {"question": question, "context": trouve_contexte(question)},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

    embeddings, index_path = change_model(settings["model"])
    cl.user_session.set("embeddings", embeddings)
    cl.user_session.set("index_path", index_path)

    # add_documents(settings["addDocuments"])
    # print(str(settings["addDocuments"])+" bien ajouté à l'index")
