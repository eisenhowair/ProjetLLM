from github_recup import *
import chainlit as cl
from chainlit.input_widget import TextInput

import sys
import os
# pour avoir accès à ce qu'il y a dans ProjetLLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
    reponse_sur_github = ask_github_index(
        cl.user_session.get("index"), question=question)

    msg = cl.Message(content=reponse_sur_github, author="connexion nwaaaar")

    await msg.send()


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    msg = cl.Message(content="Index en cours de chargement...")
    await msg.send()
    documents_repo = fetch_repository(
        repo_name=settings["name"], repo_owner=settings["owner"], repo_url=settings["url"])

    if fetch_repository == -1:
        msg.content = "Problème lors du chargement de l'index"
    else:
        index = charge_index(documents=documents_repo)

        cl.user_session.set("index", index)
        msg.content = "Index bien chargé"
    await msg.update()
