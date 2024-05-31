from github_recup import *
import chainlit as cl
from chainlit.input_widget import TextInput
import time
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine

"""
Temps pris pour génération de l'index:189.7270061969757
Temps pris pour la réponse à la question:182.3755898475647
instructor_large
( chainlit ne met pas à jour ses messages ou envoie le message de la réponse, alors qu'il est bien récupéré)


Temps pris pour génération de l'index:40.64496183395386
Temps pris pour la réponse à la question:125.97727012634277
mpnet-base-v2
"""

@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            TextInput(id="owner", label="Owner Name",
                      placeholder="https://github.com/.../repository"),
            TextInput(id="name", label="Repo Name",
                      placeholder="https://github.com/owner/..."),
            TextInput(id="url", label="Url", placeholder="..."),

        ]
    ).send()


@cl.on_message
async def main(message):
    # memory = cl.user_session.get("memory")
    question = message.content
    print("Question:" + question)
    start = time.time()
    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine

    msg = cl.Message(content="", author="connexion nwaaaar")

    res = await cl.make_async(query_engine.query)(message.content)#(create_prompt(message.content))

 
    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()

    end = time.time()
    print(f"-----\nTemps pris pour la réponse à la question:{end-start}")


@cl.step(type="run", name="Récupération de l'index")
def recup_index(settings):
    
    documents_repo = fetch_repository(
            repo_name=settings["name"], repo_owner=settings["owner"], repo_url=settings["url"])

    if fetch_repository == -1:
        print("problème pour le dépot git")
    else:
        index,service_context = charge_index(documents=documents_repo)
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=2, service_context=service_context)
        print("index bien chargé")
        cl.user_session.set("query_engine", query_engine)



@cl.on_settings_update
async def setup_agent(settings):
    start = time.time()
    recup_index(settings=settings)
    end = time.time()
    print(f"-----\nTemps pris pour génération de l'index:{end-start}")
