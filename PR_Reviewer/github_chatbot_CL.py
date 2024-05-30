from github_recup import *
import chainlit as cl
from chainlit.input_widget import TextInput
import time

import sys
import os
# pour avoir accès à ce qu'il y a dans ProjetLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

    reponse_sur_github = ask_github_index(
        cl.user_session.get("index"), question=question)

    msg = cl.Message(content=reponse_sur_github, author="connexion nwaaaar")

    await msg.send()
    end = time.time()
    print(f"-----\nTemps pris pour la réponse à la question:{end-start}")




@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    start = time.time()

    msg = cl.Message(content="Index en cours de chargement...")
    await msg.send()
    documents_repo = fetch_repository(
        repo_name=settings["name"], repo_owner=settings["owner"], repo_url=settings["url"])

    if fetch_repository == -1:
        msg.content = "Problème lors du chargement de l'index"
        print("problème pour le dépot git")
    else:
        index = charge_index(documents=documents_repo)
        print("index bien chargé")
        cl.user_session.set("index", index)
        msg.content = "Index bien chargé"
        end = time.time()
        print(f"-----\nTemps pris pour génération de l'index:{end-start}")

        await msg.update()
