import os
import chainlit as cl
from numpy import vectorize
from operator import itemgetter
from typing import List

from chainlit.types import ThreadDict
from chainlit.input_widget import TextInput, Select

from langchain_community.vectorstores import FAISS
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

from utils.manip_documents import *
from utils.web_scraper import *

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
            history=RunnableLambda(
                memory.load_memory_variables) | itemgetter("history")
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

    results_by_source = {}
    for result in search_results:
        source = result.metadata["source"]
        if source not in results_by_source:
            results_by_source[source] = []
        results_by_source[source].append(result)

    # sources différentes
    relevant_sources = list(results_by_source.keys())[:3]
    # chunks par source
    relevant_results = [results_by_source[source][:10]
                        for source in relevant_sources]

    # Aplatir la liste des résultats
    relevant_results = [
        chunk for sublist in relevant_results for chunk in sublist]

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
    await cl.ChatSettings(
        [
            TextInput(
                id="addDocuments",
                label="Précisez le chemin",
                placeholder="differents_textes/...",
            ),
            Select(
                id="model",
                label="Model",
                values=[
                    "instructor-base",
                    "instructor-large",
                    "mpnet-v2",
                    "camembert-base",
                ],
                initial_index=1,
            ),
        ]
    ).send()
    cl.user_session.set(
        "memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("nom_model", "instructor-large")

    charge_index(index_path, embeddings)


def charge_index(new_index_path, new_embeddings):
    # print(f"index_path: {new_index_path}\nembeddings:{new_embeddings}")
    if os.path.exists(new_index_path):
        vectorstore = FAISS.load_local(
            new_index_path,
            embeddings=new_embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Index chargé à partir du chemin existant.")
    else:

        # pour faire disparaitre un warning
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        webpage_dict = [
            {"url": "https://e-services.uha.fr/fr/index.html", "type": "connexion"},
            {"url": "https://www.emploisdutemps.uha.fr/", "type": "edt"},
            {
                "url": "https://e-formation.uha.fr/login/index.php?authCAS=CAS",
                "type": "connexion",
            },
            {
                "url": "https://e-formation.uha.fr/my/courses.php",
                "type": "accueil_moodle",
            },  # page mes cours sur moodle
            {"url": "https://e-partage.uha.fr/modern/email/Sent", "type": "partage"},
            {
                "url": "https://www.uha.fr/fr/formation-1/accompagnement-a-la-reussite-1/numerique.html",
                "type": "plain",
            },
        ]
        web_scraped = load_web_documents_firefox(
            webpage_dict, "https://cas.uha.fr/cas/login"
        )
        if web_scraped is None:
            "web_scraped est vide"
        chunks_web = web_scraped["web_result"]
        chunks_pdf = load_new_documents("differents_textes")

        print("les deux chunks ont été récupérés")
        print(f"liste des url pdf:{web_scraped['pdf_to_read']}")

        vectorstore = FAISS.from_documents(
            documents=chunks_web + chunks_pdf, embedding=new_embeddings
        )

        vectorstore.save_local(new_index_path)
        print("Nouvel index créé et sauvegardé.")

    print(f"Type d'index: {vectorstore.index}")
    retriever = vectorstore.as_retriever()
    cl.user_session.set("retriever", retriever)


@cl.on_message
async def main(message):
    memory = cl.user_session.get("memory")
    question = message.content
    print("Question:" + question)

    # setup_model() et trouve_contexte() à adapter suivant ce qui est recherché
    runnable_model = setup_model()
    msg = cl.Message(content="", author=cl.user_session.get("nom_model"))
    async for chunk in runnable_model.astream(
        {"question": question, "context": trouve_contexte(question)},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(msg.content)


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

    embeddings_t, index_path_t = change_model(settings["model"])
    charge_index(index_path_t, embeddings_t)
    cl.user_session.set("nom_model", settings["model"])
    if settings["addDocuments"] is not None:
        add_files_to_index(index_path_t, embeddings_t,
                           settings["addDocuments"])
        settings["addDocuments"] = ""


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    await factory()
